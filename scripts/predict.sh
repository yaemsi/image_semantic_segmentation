#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export VLLM_DISABLE_COMPILE_CACHE=1


# Parameters
export MODEL_PATH="./output_image_segmentation/final"
export DS_PATH="./data/raw/test"
export OUTPUT_DIR="./output_prediction"
export BATCH_SIZE=32
export N_PROC=4


python  main.py --predict \
    --model_path $MODEL_PATH \
    --img_path $DS_PATH \
    --save_path $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_proc $N_PROC

