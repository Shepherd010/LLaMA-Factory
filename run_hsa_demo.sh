#!/bin/bash

export USE_HSA=true

# Use Qwen2-VL-2B-Instruct as the model (downloaded locally)
MODEL_PATH="models/Qwen2-VL-2B-Instruct"

# Output directory
OUTPUT_DIR="saves/hsa_demo"

# Run training
conda run -n llama-factory llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path $MODEL_PATH \
    --dataset hsa_test \
    --dataset_dir data \
    --template qwen2_vl \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 10 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --max_samples 100 \
    --max_steps 10 \
    --fp16 \
    --remove_unused_columns False \
    --flash_attn disabled \
    --plot_loss \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen \
    --optim adamw_torch 
