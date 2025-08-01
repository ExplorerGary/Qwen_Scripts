import torch
import os
import torch.distributed as dist
import numpy as np

# --- helper function ---
def quantlization_fuct(flat_tensor:torch.Tensor,
                       scaling:float = None,
                       fp64_enable:bool = False):
    '''
    观察记录：
    1. fp16的最高数字约为为6.5e5，也就意味着我们最好不要使用1e3及以上的tensor，不然就变成inf了(因为有情况下时会出现Xe2的数量级的)
        但不知道为什么，经过测试后发现原来的scaling是可行的。
    
    '''
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
    '''
    由mod_allreduce_hook_base修改而来：
    本体允许结合名为EPOCH_STEP_HANDLER的TrainerCallback，实现：
        1. 知晓这个bucket的meta数据 {
            1. rank
            2. epoch
            3. step
            4. index
        }
        2. 记录每一个bucket里的内容
        3. 根据save_Bucket变量决定是否保存GradBucket里面的数据
    
    更新后支持：
        1. 使用一个scaling参数对数据进行quantlization -- 2025年7月29日实现
    '''
    
    
    # --- 导入东西 --- 
    global CURRENT_EPOCH,CURRENT_STEP,save_Bucket,Scaling,param_name_map,OUTPUT_DIR,Pioneer
    
    # --- 缓冲区里的扁平向量 --- 
    flat_tensor = bucket.buffer()
    
    # --- 基本信息 --- 
    # 1. 知道这个是哪个rank:
    rank = dist.get_rank()
    
    # 2. 知道这是这个batch(或者step)第几个bucket:
    idx = bucket.index()
    
    # 3. 知道存储的数据类型：
    data_type = flat_tensor.dtype
    
    # 4. 知道这个桶里面塞了什么？然后存下来！
    params = bucket.parameters()  # List[Tensor]
    grads = bucket.gradients()  # List[Tensor]，对应顺序应该和 params 一致 -- [已确认]

    
    # 4.1 知道这个桶属于哪个step和epoch
    the_epoch = CURRENT_EPOCH
    the_step = CURRENT_STEP

    
    #### DEBUGING ####
    if Pioneer:
        print(f"HOOK TRIGGERED: rank {rank}, epoch {the_epoch}, step {the_step}, bucket_idx {idx}, dtype = {data_type}")
    ##################
    
    
    ### 更新 ###
    # 1. 量化
    if Scaling is not None:
        quantized = quantlization_fuct(flat_tensor=flat_tensor,scaling=Scaling,fp64_enable=False)
                                       
    # 2. val2index
    
    
    
    # 2.1 val2bucket
    # 2.2 bucket2index
        # 3. EG Encoding
    
    

    
    
    # bucket.set_buffer(codes) # 将bucket的内容更改为EG encoding的结果: codes
    
    ############
    
    
    
    
    # 4.1.1:
    
    # 文件名称：
    file_name = f"R_{rank}_E_{the_epoch}_S_{the_step}_B_{idx}.pt"
    # 保存路径

    os.makedirs(OUTPUT_DIR,exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, file_name)
    

    
    # 4.2 具体保存
    try:
        param_names = [param_name_map.get(id(p), "UNKNOWN_PARAM") for p in params]

        # print("save_bucket",save_Bucket)
        if save_Bucket:
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
    if the_epoch == 0 or 1: # 只保存前两个epoch的debug信息
        to_path = os.path.join(OUTPUT_DIR,"000_EG_Full_DEBUG_INFO_{rank}.txt")
        with open(to_path,"a") as DEBUG_FILE:
            DEBUG_FILE.write(INFO)
    

    # --- 原本的逻辑 ---
    return _allreduce_fut(process_group, bucket.buffer())