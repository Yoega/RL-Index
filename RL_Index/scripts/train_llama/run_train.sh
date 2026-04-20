#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1,2,3
accelerate launch --config_file=scripts/train_llama/zero2.yaml \
    scripts/train_llama/train_grpo.py \
    --training_model_path meta-llama/Llama-3.2-3B-Instruct \
    --train_data_path data_process/data/built_dataset/v2_train \
    --scoring_model_path sentence-transformers/all-mpnet-base-v2 \
    --per_device_train_batch_size 4 \
    --grad_accum_steps 16 \
    --num_generations 16 \
    --max_seq_length 8000 \
    --max_completion_length 500 \
    --max_steps 1500 \
    --learning_rate 1e-6 \
    --beta 0.008 \
    --seed 5202 \
    --output_dir outputs \
