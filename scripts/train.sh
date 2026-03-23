#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export VLLM_DISABLE_COMPILE_CACHE=1
export WANDB_API_KEY=""


# Debug
export CUDA_LAUNCH_BLOCKING=1 
export TORCH_USE_CUDA_DSA=1

# Parameters
export ACC_CFG="./configs/acc_no_dynamo.yaml"
export DS_PATH="./data/processed"
export OUTPUT_DIR="./output_image_segmentation"
export LR=6e-5
export MAX_STEPS=25000
export WARMUP_STEPS=8000
export EVAL_STEPS=2500
export BATCH_SIZE=32


accelerate launch --config_file $ACC_CFG main.py --train \
    --dataset_path $DS_PATH \
    --output_dir $OUTPUT_DIR \
    --learning_rate $LR \
    --max_steps $MAX_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --save_steps $EVAL_STEPS \
    --eval_steps $EVAL_STEPS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE


