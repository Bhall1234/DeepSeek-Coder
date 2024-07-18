#!/bin/bash
#SBATCH --partition=amp48         
#SBATCH --qos=amp48              
#SBATCH --gpus 4              
#SBATCH -c8 -- mem=64gb

# Load necessary modules
#module load conda
#conda activate deepspeed_env

# Navigate to the repository directory
#cd $SLURM_SUBMIT_DIR

# Run the finetuning script
deepspeed finetune/finetune_deepseekcoder.py \
    --model_name_or_path deepseek-ai/deepseek-coder-1.3b-instruct \
    --data_path /path/to/your/data.json \
    --output_dir /path/to/output_dir \
    --num_train_epochs 3 \
    --model_max_length 1024 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --deepspeed configs/ds_config_zero3.json \
    --bf16 True