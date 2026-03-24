#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export VLLM_DISABLE_COMPILE_CACHE=1


# Parameters
export MODEL_PATH="./output_image_segmentation/final"
export DS_PATH="./data/processed"
export OUTPUT_DIR="./output_evaluation"
export SPLIT="test"
export BATCH_SIZE=32
export N_PROC=2


python  main.py --evaluate \
    --model_path $MODEL_PATH \
    --ds_path $DS_PATH \
    --save_path $OUTPUT_DIR \
    --split $SPLIT \
    --batch_size $BATCH_SIZE \
    --num_proc $N_PROC
