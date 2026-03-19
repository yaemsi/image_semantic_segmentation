#!/bin/bash

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export VLLM_DISABLE_COMPILE_CACHE=1
export WANDB_API_KEY="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"


# Debug
export CUDA_LAUNCH_BLOCKING=1 
export TORCH_USE_CUDA_DSA=1

#python main.py --train \

accelerate launch --config_file ./configs/acc_no_dynamo.yaml \
    main.py --train \
    --dataset_path ./data/processed \
    --output_dir ./output_image_segmentation \
    --learning_rate 6e-5 \
    --max_steps 25000 \
    --warmup_steps 8000 \
    --save_steps 2500 \
    --eval_steps 2500 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32


