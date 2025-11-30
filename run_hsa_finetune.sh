#!/bin/bash

# Set environment variable to enable HSA
export USE_HSA=true
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Run LoRA Fine-tuning
CUDA_VISIBLE_DEVICES=0 python src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path models/Qwen2-VL-2B-Instruct \
    --dataset hsa_test \
    --dataset_dir data \
    --template qwen2_vl \
    --finetuning_type lora \
    --lora_target all \
    --output_dir saves/Qwen2-VL-2B-Instruct/lora/hsa_demo \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 100 \
    --learning_rate 1e-4 \
    --num_train_epochs 3.0 \
    --max_samples 100 \
    --plot_loss \
    --fp16
