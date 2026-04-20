#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1,2,3
export NCCL_P2P_DISABLE=1      
export NCCL_SHM_DISABLE=1      
export NCCL_IB_DISABLE=1
accelerate launch --config_file=scripts/train/zero2.yaml \
    scripts/train_qwen_sbert/train_grpo.py \
    --training_model_path Qwen/Qwen2.5-1.5B-Instruct \
    --train_data_path data_process/data/built_dataset/v2_train_no_think \
    --per_device_train_batch_size 8 \
    --grad_accum_steps 8 \
    --num_generations 16 \
    --max_seq_length 8000 \
    --max_completion_length 500 \
    --max_steps 1000 \
    --learning_rate 1e-6 \
    --beta 0.008 \
    --seed 5202 \
    --output_dir outputs \
    --log_file logs/training_seed5202.log \
