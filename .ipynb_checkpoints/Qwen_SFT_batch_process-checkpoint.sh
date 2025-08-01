#!/bin/bash

### CONFIGS ###

SCRIPT_PATH_FULL="/gpfsnyu/scratch/zg2598/Qwen/Qwen_Scripts/Qwen_Full_SFT_test_quant.py"
SAVE_BUCKET=false
SCALING_VALUES=(256 1e3 1e4 None) # fp16无法承受1e5的scaling，会出事的
NPROC_PER_NODE=2
USE_PIONEER=false

# 限制线程数量，避免 PyTorch 警告
export OMP_NUM_THREADS=1

echo "RUNNING Full_SFT...."

SCRIPT_PATH="$SCRIPT_PATH_FULL"  

for scaling in "${SCALING_VALUES[@]}"; do
    echo "==============================="
    echo "Starting training with scaling=${scaling}"

    CMD="torchrun --nproc_per_node=${NPROC_PER_NODE} ${SCRIPT_PATH}"

    if [[ "$scaling" != "None" ]]; then
        CMD+=" --scaling ${scaling}"
        CMD+=" --scaling_str ${scaling}"
    fi

    if [[ "$USE_PIONEER" == true ]]; then
        CMD+=" --pioneer"
    fi

    if [[ "$SAVE_BUCKET" == true ]]; then
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
