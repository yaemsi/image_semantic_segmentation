#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_DISABLE_COMPILE_CACHE=1


python  main.py --train \
    --dataset_path ./data/processed \
    --output_dir ./output_image_segmentation \
    --learning_rate 6e-5 \
    --max_steps 25000 \
    --warmup_steps 8000 \
    --save_steps 2500 \
    --eval_steps 2500 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32