#!/bin/bash

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export VLLM_DISABLE_COMPILE_CACHE=1
export WANDB_API_KEY="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"


#python main.py --train \

accelerate launch --config_file ./configs/acc_no_dynamo.yaml \
    main.py --train \
    --dataset_path ./data/processed \
    --output_dir ./output_image_segmentation \
    --learning_rate 6e-5 \
    --max_steps 50000 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32


