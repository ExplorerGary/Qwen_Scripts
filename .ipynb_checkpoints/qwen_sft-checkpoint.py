import os
import torch
import pandas as pd
import transformers
import peft
import accelerate
import datasets
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
import torch.distributed as dist
import argparse



# 量化函数
def quantlization_fuct(flat_tensor:torch.Tensor,
                       scaling:float = None,
                       fp64_enable:bool = False):
    global Pioneer
    if Pioneer:
        print(f"doing quantlization, scaling = {scaling}")
    try:
        if fp64_enable:
            flat_tensor = flat_tensor.to(dtype=torch.float64)
        quantilized = torch.round(flat_tensor * scaling) / scaling
        if scaling is None:
            quantilized = flat_tensor
        return quantilized
    except Exception as e:
        raise e    

print(f"local rank: {int(os.environ.get('LOCAL_RANK', 0))}")

# 加载模型和分词器
model_name = "/gpfsnyu/home/zg2598/Qwen"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 加载数据集
dataset_path = "/gpfsnyu/home/zg2598/datasets/medical_sft/"
ds = load_from_disk(dataset_path)
ds = ds["train"]  # load all the data

def formating_function(sample):
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant, aiming to provide medical advice to those in need."},
        {"role": "user", "content": sample["Question"]},
        {"role": "assistant", "content": sample["Response"]}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": formatted_prompt}

# 数据局处理完毕
train_ds = ds.map(formating_function, remove_columns=ds.column_names)

hooked_args = SFTConfig(
    output_dir="/gpfsnyu/home/zg2598/Qwen/OUT",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    warmup_steps=5,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=5,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=42,
    report_to="tensorboard",
    # fp16=True,
    bf16=True,
    max_grad_norm=1.0,
    logging_first_step=True,
    save_steps=0,
    save_strategy="epoch"
)

from transformers import TrainerCallback
class EPOCH_STEP_HANDLER(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        global CURRENT_EPOCH
        CURRENT_EPOCH = int(state.epoch or 0)
        return super().on_epoch_begin(args, state, control, **kwargs)
    
    def on_step_begin(self, args, state, control, **kwargs):
        global CURRENT_STEP
        CURRENT_STEP = state.global_step
        return super().on_step_begin(args, state, control, **kwargs)

from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook 

def _allreduce_fut(process_group: dist.ProcessGroup, tensor: torch.Tensor) -> torch.futures.Future[torch.Tensor]:
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    tensor.div_(group_to_use.size())
    return (
        dist.all_reduce(tensor, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )

def mod_allreduce_hook_EG(process_group: dist.ProcessGroup, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    global CURRENT_EPOCH, CURRENT_STEP, save_Bucket, Scaling, param_name_map, OUTPUT_DIR

    flat_tensor = bucket.buffer()
    rank = dist.get_rank()
    idx = bucket.index()
    params = bucket.parameters()
    grads = bucket.gradients()
    the_epoch = CURRENT_EPOCH
    the_step = CURRENT_STEP

    print(f"HOOK TRIGGERED: rank {rank}, epoch {the_epoch}, step {the_step}, bucket_idx {idx}")

    if Scaling is not None:
        quantized = quantlization_fuct(flat_tensor=flat_tensor, scaling=Scaling, fp64_enable=False)
        bucket.set_buffer(quantized)

    file_name = f"R_{rank}_E_{the_epoch}_S_{the_step}_B_{idx}.pt"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, file_name)

    try:
        param_names = [param_name_map.get(id(p), "UNKNOWN_PARAM") for p in params]
        if save_Bucket:
            grad_dict = {}
            for name, grad_tensor in zip(param_names, grads):
                if grad_tensor is not None:
                    grad_dict[name] = grad_tensor
                else:
                    print(f"[Rank {rank}] WARNING: Gradient for {name} is None")
                pass
            torch.save(grad_dict, save_path)
    except Exception as e:
        print(f"[Rank {rank}] Error accessing bucket parameters: {e}")
        param_names = "ERROR!!!"

    INFO = f"""
===========
[INFO]
rank: {rank}
epoch: {the_epoch}
step: {the_step}
bucket_idx: {idx}
    ---
contents:
{param_names}
===========
"""
    if the_epoch in (0, 1):
        to_path = os.path.join(OUTPUT_DIR, f"000_EG_Full_DEBUG_INFO_{rank}.txt")
        with open(to_path, "a") as DEBUG_FILE:
            DEBUG_FILE.write(INFO)

    return _allreduce_fut(process_group, bucket.buffer())

class HookedSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.communication_data = []
        self.hook_registered = False
        self.param_name_map = None
        self.epoch_step_config_0 = None
        self.epoch_step_config_1 = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.hook_registered == False:
            print(f"Hooked??? --- {self.hook_registered}")
        if dist.is_initialized() and self.hook_registered == False:
            try:
                global param_name_map, epoch_step_config_0, epoch_step_config_1
                param_name_map = {id(p): name for name, p in model.named_parameters()}
                self.param_name_map = param_name_map
                epoch_step_config_0 = {"epoch": 0, "step": 0}
                self.epoch_step_config_0 = epoch_step_config_0
                epoch_step_config_1 = {"epoch": 0, "step": 0}
                self.epoch_step_config_1 = epoch_step_config_1
                print("config initiallized!!!")
                model.register_comm_hook(state=None, hook=mod_allreduce_hook_EG)
                self.hook_registered = True
                print("HOOKED!!!")
            except Exception as e:
                print(f"Something bad happened: {e}")
        return super().training_step(model, inputs, num_items_in_batch)

def main(save_bucket=False, scaling=None, pioneer=False, output_dir_name=None):
    global save_Bucket, Scaling, Pioneer, OUTPUT_DIR

    save_Bucket = save_bucket
    Scaling = scaling
    Pioneer = pioneer
    print(f"SAVING BUCKET???\n--{save_Bucket}")

    # 准备数据集，假设你有prepare_dataset函数，如果没有请定义或直接用train_ds或subset
    # 这里示范使用之前准备好的 train_ds 子集
    subset = train_ds.train_test_split(test_size=11/12, seed=42)["train"]
    dataset = subset if pioneer else train_ds # 你也可以根据pioneer控制加载不同子集

    if output_dir_name is None:
        output_dir_name = "result_None"
    BASE_RESULT_DIR = "/gpfsnyu/home/zg2598/Qwen"  # 你自己定义基路径
    save_dir = os.path.join(BASE_RESULT_DIR, f"result_{output_dir_name}")
    OUTPUT_DIR = os.path.join(save_dir, "COMMUNICATION_LOG")

    args = SFTConfig(
        output_dir=save_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=5,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        push_to_hub=False,
        report_to="tensorboard",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    args.remove_unused_columns = False
    print(f"results will be saved to\n{args.output_dir}")

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=int(os.environ.get("LOCAL_RANK", 0)),
        trust_remote_code=True
    )
    model.enable_input_require_grads()

    # 你这里的processor和collate_fn没定义，假设用tokenizer和默认collator替代
    processor = tokenizer
    collate_fn = None  # 如果你有自定义collate函数请替换

    # 初始化训练器
    hooked_trainer = HookedSFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=processor,
        data_collator=collate_fn,
        callbacks=[EPOCH_STEP_HANDLER()]
    )

    print("Training begin...")
    hooked_trainer.train()

    # 训练结束后销毁进程组
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",  # 你的设备支持的backend
            init_method="env://",
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaling", type=float, default=None, required=False)
    parser.add_argument("--save_bucket", action="store_true", default=False)
    parser.add_argument("--pioneer", action="store_true", default=False)
    parser.add_argument("--scaling_str", type=str, required=False, default=None)
    args_ = parser.parse_args()

    save_bucket = args_.save_bucket
    scaling = args_.scaling
    pioneer = args_.pioneer
    output_dir_name = args_.scaling_str

    main(save_bucket=save_bucket, scaling=scaling, pioneer=pioneer, output_dir_name=output_dir_name)








