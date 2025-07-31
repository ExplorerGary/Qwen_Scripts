#!/bin/bash


### CONFIGS ###

# 设置你的训练脚本路径
SCRIPT_PATH_FULL="/gpfsnyu/scratch/zg2598/Qwen/Qwen_Scripts/qwen_sft.py"

# 是否要保存通信文件
SAVE_BUCKET=false

# 设置你要尝试的 scaling 值数组（可以根据需要添加或修改）
SCALING_VALUES=(256 1e3 1e4 1e5 None)

# 设置每次训练使用的 GPU 数（nproc）
NPROC_PER_NODE=2

# pioneer 模式是否启用（true/false）
USE_PIONEER=true


############################################


echo "RUNNING Full_SFT...."

# ✅ 选择当前要使用的脚本
SCRIPT_PATH="$SCRIPT_PATH_FULL"  

# 遍历所有 scaling 值
for scaling in "${SCALING_VALUES[@]}"; do
    echo "==============================="
    echo "Starting training with scaling=${scaling}"

    # 生成命令
    CMD="torchrun --nproc_per_node=${NPROC_PER_NODE} ${SCRIPT_PATH}"

    # 如果 scaling 不是 None，就加参数
    if [[ "$scaling" != "None" ]]; then
        CMD+=" --scaling ${scaling}"
        CMD+=" --scaling_str ${scaling}"
    fi

    # 如果启用 pioneer 模式
    if [[ "$USE_PIONEER" == true ]]; then
        CMD+=" --pioneer"
    fi

    if [["$SAVE_BUCKET" == true]]; then
        CMD+=" --save_bucket"
    fi

    echo "Running command:"
    echo "$CMD"
    eval "$CMD"

    echo "Finished run with scaling=${scaling}"
    echo "==============================="
    echo ""
done


echo "ALL COMPLETED..."