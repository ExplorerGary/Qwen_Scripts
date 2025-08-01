'''the below is for HPC'''
# Activate the correct environment and run this script with torchrun for distributed training.
# Example command (adjust --nproc_per_node as needed for your GPUs):

# torchrun --nproc_per_node=NUM_GPUS_YOU_HAVE /D:/NYU_Files/2025\ SPRING/Summer_Research/新/PYTHON/hook_for_larger_llms/qwen_sft_hooked.py

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
from transformers import TrainerCallback
import time
import csv


BASE_RESULT_DIR = "/gpfsnyu/scratch/zg2598/Qwen/"

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



print(f"local rank: {int(os.environ.get("LOCAL_RANK", 0))}")
model_name = "/gpfsnyu/home/zg2598/Qwen"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


# Load dataset
dataset_path = "/gpfsnyu/home/zg2598/datasets/medical_sft/"

def formating_function(sample):
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant, aiming to provide medical advice to those in need."},
        {"role": "user", "content": sample["Question"]},
        {"role": "assistant", "content": sample["Response"]}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": formatted_prompt}

def prepare_dataset(pioneer=False):
    ds = load_from_disk(dataset_path)
    ds = ds["train"] 
    train_ds = ds.map(formating_function, remove_columns=ds.column_names)
    subset = train_ds.train_test_split(test_size=11/12, seed=42)["train"] 
    
    if pioneer:
        return subset
    
    return train_ds


    
""" --- self_defined SFTTrainer ---"""
# EPOCH 和 STEP 怎么找：
def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d

class EPOCH_STEP_HANDLER(TrainerCallback):
# class EPOCH_STEP_HANDLER(DefaultFlowCallback):
    def __init__(self, filename="training_log.csv"):
        self.filename = filename
        self.fieldnames = None
        self.csv_file = None
        self.writer = None
        self.csv_path = None
        
    def is_main_process(self):
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0
    def on_epoch_begin(self, args, state, control, **kwargs):
        """
        Event called at the beginning of an epoch.
        """
        global CURRENT_EPOCH
        CURRENT_EPOCH = int(state.epoch or 0)
        
        return super().on_epoch_begin(args, state, control, **kwargs)
        
    
    def on_step_begin(self, args, state, control, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        global CURRENT_STEP
        CURRENT_STEP = state.global_step
        
        return super().on_step_begin(args, state, control, **kwargs)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.is_main_process():
            return  # ⛔ 非主进程不写日志

        try:
            if logs is None:
                return

            # ✅ 你要求的固定开头部分
            logs = rewrite_logs(logs)
            the_output_dir = args.output_dir
            csv_dir = os.path.join(the_output_dir, "TRAINING_LOG")
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(csv_dir, self.filename)
            self.csv_path = csv_path
            # 当前 step 的字段
            current_keys = ["step", "time"] + list(logs.keys())

            # 初始化 CSV 写入器
            if self.writer is None:
                self.csv_file = open(csv_path, mode="w", newline="")
                self.fieldnames = current_keys
                self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
                self.writer.writeheader()

            # 写入当前 row
            row = {
                "step": state.global_step,
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }
            for key in self.fieldnames:
                if key not in row:  # 跳过 step 和 time，因为已经加了
                    row[key] = logs.get(key, "")
            self.writer.writerow(row)
            self.csv_file.flush()

        except Exception as e:
            print(f"CSV Logging Error: {e}")

    def on_train_end(self, args, state, control, **kwargs):
        if self.csv_file:
            self.csv_file.close()
        print(f"CSV SAVED TO {self.csv_path}!")

# DDP钩子：
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook 
#           mod_allreduce_hook: 添加读取和保存的信息：

# --- helper function ---
def _allreduce_fut(
    process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future[torch.Tensor]:
    """Average the input gradient tensor by allreduce and returns a future."""
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    # Apply the division first to avoid overflow, especially for FP16.
    tensor.div_(group_to_use.size())

    return (
        dist.all_reduce(tensor, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )
    
# --- hook本体 ---
def mod_allreduce_hook_EG(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    # --- 导入东西 --- 
    global CURRENT_EPOCH,CURRENT_STEP,param_name_map,Pioneer
    # --- 缓冲区里的扁平向量 --- 
    flat_tensor = bucket.buffer()
    # --- 基本信息 --- 
    output_path = "/gpfsnyu/scratch/zg2598/Qwen/OUT/COMMUNICATION_LOG/"
    
    # 1. 知道这个是哪个rank:
    rank = dist.get_rank()
    
    # 2. 知道这是这个batch(或者step)第几个bucket:
    idx = bucket.index()
    
    # 3. 知道存储的数据类型：
    data_type = flat_tensor.dtype
    
    
    #### DEBUGING ####
    if Pioneer:
        print(f"HOOK TRIGGERED: rank {rank}, epoch {the_epoch}, step {the_step}, bucket_idx {idx}, dtype = {data_type}")
    ##################
    
    # 4. 知道这个桶里面塞了什么？然后存下来！
    params = bucket.parameters()  # List[Tensor]
    grads = bucket.gradients()  # List[Tensor]，对应顺序应该和 params 一致 -- [已确认]

    
    # 4.1 知道这个桶属于哪个step和epoch
    the_epoch = CURRENT_EPOCH
    the_step = CURRENT_STEP
    
    
    ### EG 相关 ###
    # 1. 量化
    if Scaling is not None:
        quantized = quantlization_fuct(flat_tensor=flat_tensor,
                                       scaling=Scaling,
                                       fp64_enable=False)
        # set_buffer
        bucket.set_buffer(quantized) # 2025年7月29日：测试量化后的表现 
    # 2. val2index
    
    # 3. EG Encoding
    
    

    
    
    # bucket.set_buffer(codes) # 将bucket的内容更改为EG encoding的结果: codes
    
    ############
    
    # 4.1.1:
    
    # 文件名称：
    file_name = f"R_{rank}_E_{the_epoch}_S_{the_step}_B_{idx}.pt"
    # 保存路径
    save_path = os.path.join(output_path, file_name)
    
    # 4.2 具体保存
    try:
        param_names = [param_name_map.get(id(p), "UNKNOWN_PARAM") for p in params]

        grad_dict = {}
        for name,grad_tensor in zip(param_names,grads):            
            # 将这个bucket的所有grad，按照name:grad_tensor的键对值形式保存进一个.pt文件里，日后备用
            # pt_file_name = f"R_{rank}_E_{epoch}_S_{step}_B_{idx}.pt"
            if grad_tensor is not None:
                grad_dict[name] = grad_tensor  # .cpu()  # 先转 cpu，避免 GPU 阻塞
            else: # 一般情况下不会发生
                print(f"[Rank {rank}] WARNING: Gradient for {name} is None")
            pass
        
        torch.save(grad_dict, save_path) # 分开保存
        # torch.save(flat_tensor,save_path) # 整体保存
            
    except Exception as e:
        print(f"[Rank {rank}] Error accessing bucket parameters: {e}")
        param_names = "ERROR!!!"
        
        
    # 保存调试信息：
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
    if the_epoch == 0 or 1: # 只保存前两个epoch的东西
            
        with open(f"{output_path}000_DEBUG_INFO_{rank}.txt","a") as DEBUG_FILE:
            DEBUG_FILE.write(INFO)
    

    # --- 原本的逻辑 ---
    return _allreduce_fut(process_group, bucket.buffer())

class HookedSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.communication_data = []  # Store communication data
        self.hook_registered = False  # Track hook registration
        self.param_name_map = None
        
        # 一定有更好的方法解决这个问题
        self.epoch_step_config_0 = None
        self.epoch_step_config_1 = None
        
    def training_step(
        self, model, inputs, num_items_in_batch=None
    ):
        # input args: model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        # --- DDP 钩子 ---
        if self.hook_registered == False:
            print(f"Hooked??? --- {self.hook_registered}")


        # Make sure allreduce_hook is defined or imported before using it
        # from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
        if dist.is_initialized() and self.hook_registered == False:
            try:
                global param_name_map
                global epoch_step_config_0
                global epoch_step_config_1
                
                param_name_map = {id(p): name for name, p in model.named_parameters()}
                self.param_name_map = param_name_map
                
                epoch_step_config_0 =  {"epoch":0,"step":0}   
                self.epoch_step_config_0 = epoch_step_config_0
                
                epoch_step_config_1 = {"epoch":0,"step":0}
                self.epoch_step_config_1 = epoch_step_config_1
                
                print("config initiallized!!!")
                
                model.register_comm_hook(state=None, hook=mod_allreduce_hook_EG)
                self.hook_registered = True
                print("HOOKED!!!")
            except Exception as e:
                print(f"Something bad happened: {e}")

        # ---调用本家的东西 --- 
        return super().training_step(model,inputs,num_items_in_batch)



def run_train(save_bucket=False,
              scaling = None,
              output_dir_name = None,
              pioneer = False):
    
    global save_Bucket, Scaling, Pioneer, OUTPUT_DIR

    save_Bucket = save_bucket
    Scaling = scaling
    Pioneer = pioneer
    print(f"SAVING BUCKET???\n--{save_Bucket}")
    
    if output_dir_name is None:
        output_dir_name = "None"
    save_dir = os.path.join(BASE_RESULT_DIR,f"result_Lora_{output_dir_name}")
    global OUTPUT_DIR 
    OUTPUT_DIR = os.path.join(save_dir,"COMMUNICATION_LOG")
    
    
    hooked_args = SFTConfig(
    output_dir=f"/gpfsnyu/home/zg2598/Qwen/{output_dir_name}",
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
    report_to="none",
    # fp16=True,
    bf16=True,
    max_grad_norm=1.0,
    logging_first_step=True,
    save_steps = 0, # saving nothing
    save_strategy = "no"
    )

    
    
    hooked_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=int(os.environ.get("LOCAL_RANK", 0)),
    trust_remote_code=True
    )
    hooked_model.enable_input_require_grads()
    train_ds = prepare_dataset(pioneer=pioneer)
    sft_trainer = HookedSFTTrainer(
        model=hooked_model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        # train_dataset=subset, # 测试时只使用1/12的数据
        args=hooked_args,
        callbacks = [EPOCH_STEP_HANDLER()]
    )

    sft_trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaling", type=float, default=None, required=False)
    parser.add_argument("--save_bucket", action="store_true", default=False)
    parser.add_argument("--pioneer", action="store_true", default=False)
    args_ = parser.parse_args()
    save_bucket = args_.save_bucket
    scaling = args_.scaling
    pioneer = args_.pioneer
    output_dir_name = args_.scaling_str
    
    run_train(save_bucket=save_bucket,
              scaling=scaling,
              output_dir_name=output_dir_name,
              pioneer=pioneer)




# 有关Hooked Trainer
        # --- 发现 ---
        # 经过试验，明确 self.model_wrapped才是我们需要处理的东西，用这个注册DDP钩子准备抓取数据！
        # if dist.is_initialized() and dist.get_rank() == 0:
        #     print(f"self.model type in training_step: {type(self.model)}")
        #     print(f"self.model_wrapped type in training_step: {type(self.model_wrapped)}") # 已知这个才是我们要找的对象。
        #     # print(self.model == model)
        #     # print(self.model_wrapped == model)
        # 因此，_wrap_model就没必要修改了