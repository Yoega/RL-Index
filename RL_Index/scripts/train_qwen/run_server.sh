#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1      
export NCCL_SHM_DISABLE=1      
export NCCL_IB_DISABLE=1
MODEL_PATH='Qwen/Qwen2.5-1.5B-Instruct' # Qwen/Qwen2.5-1.5B-Instruct, meta-llama/Llama-3.2-3B-Instruct, Qwen/Qwen2.5-7B-Instruct
trl vllm-serve --model $MODEL_PATH --max_model_len 8000 --port 8000