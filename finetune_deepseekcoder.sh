#!/bin/bash

# Define paths
DATA_PATH="/home/ben/Desktop/DeepSeek-Coder/incorrect_responses_evolution_1.json"
OUTPUT_PATH="/home/ben/Desktop/DeepSeek-Coder/output"
MODEL="deepseek-ai/deepseek-coder-1.3b-instruct"

# Change directory to finetune
cd finetune

# RUN 1 (ORIGINAL/UNEDITED)
# per_device_train_batch_size was 16
# gradient_accumulation_steps was 4
# at the bottom   --bf16 True
# logging_steps was 100
# save_total_limit was 100

# RUN 2
# per_device_train_batch_size was 16
# gradient_accumulation_steps was 4
# at the bottom   --bf16 True
# logging_steps was 100
# save_total_limit was 100

# Run the finetuning script with DeepSpeed
deepspeed finetune_deepseekcoder.py \
    --model_name_or_path $MODEL \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 3 \
    --model_max_length 2048 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 10 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --deepspeed configs/ds_config_zero3.json \
    --bf16 True
