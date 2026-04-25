# RL-Index: Reinforcement Learning for Retrieval Index Reasoning

An end-to-end pipeline for enhancing retrieval through reinforcement learning-based index reasoning.

## Overview

RL-Index uses reinforcement learning techniques to improve document retrieval by training models to perform intelligent index reasoning. This repository contains a complete, production-ready pipeline from data preparation through evaluation, supporting multiple embedding models and evaluation benchmarks.

**Key Features:**
- 🚀 End-to-end training and evaluation pipeline
- 🧠 Multiple embedding model support (BGE, E5, GTE-Qwen, MPNET)
- 📊 Comprehensive evaluation metrics (NDCG, Recall@K)
- ⚡ FAISS-based efficient retrieval indexing
- 🔄 Document reasoning and augmentation with RL
- 🏗️ Distributed training with vLLM


## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Training Data Preparation](#training-data-preparation)
4. [Evaluation Data Preparation](#evaluation-data-preparation)
5. [Model Training](#model-training)
6. [Document Reasoning & Augmentation](#document-reasoning--augmentation)
7. [Embedding and Indexing](#embedding-and-indexing)
8. [Evaluation](#evaluation)
9. [Troubleshooting](#troubleshooting)
10. [Citation](#citation)

---

## Requirements

### System Requirements
- **Python:** 3.10 or higher
- **CUDA:** 12.x (NVIDIA CUDA Runtime)
- **GPU VRAM:**
  - 8GB minimum (for inference)
  - 24GB+ recommended (for model training)
- **Disk Space:** 100GB+ (for datasets and embeddings)

### Python Packages

#### Core Dependencies
- **PyTorch:** torch==2.6.0, torchaudio==2.6.0, torchvision==0.21.0
- **Transformers:** transformers==4.57.3
- **Language Models & Inference:**
  - vllm==0.8.4
  - outlines==0.1.11
  - trl==0.23.0
  - llguidance==0.7.30

#### Data & ML Libraries
- **Data Processing:** pandas==2.3.3, datasets==4.4.2, pyarrow==22.0.0
- **ML Framework:** scikit-learn==1.7.2, scipy==1.15.3
- **Numerical Computing:** numpy==1.26.4, numba==0.61.2
- **Embeddings & Retrieval:**
  - sentence-transformers==4.0.1
  - faiss-gpu-cu12==1.10.0

#### Training & Optimization
- **Training Tools:**
  - accelerate==1.12.0
  - deepspeed==0.17.5
  - cloudpickle==3.1.2

#### API & Integration
- **LLM APIs:** anthropic==0.76.0, openai==2.14.0, mistral-common==1.8.8
- **Web Framework:** fastapi==0.128.0, uvicorn==0.40.0, starlette==0.50.0
- **Monitoring:** wandb==0.21.4, sentry-sdk==2.48.0

#### CUDA/GPU Support
- nvidia-cuda-runtime-cu12==12.4.127
- nvidia-cublas-cu12==12.4.5.8
- nvidia-cudnn-cu12==9.1.0.70
- nvidia-nccl-cu12==2.21.5

#### Additional Tools
- **Utilities:** pydantic==2.12.5, pyyaml==6.0.3, click==8.3.1, rich==14.2.0
- **Development:** ipython==8.37.0, jupyter-client==8.8.0, ipykernel==7.2.0

See [requirements.txt](requirements.txt) for the complete package list with versions.

## Installation

### Basic Setup

```bash
# Clone the repository
git clone https://github.com/Yoega/RL-Index.git
cd RL-Index

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Training Data Preparation

### Overview
Prepare training data from the TongSearch-QR benchmark for model fine-tuning. This step is essential for training the RL-based index reasoning model.

### Steps

1. **Download Data**
   - Download the training dataset from: [TongSearch-QR V2 Dataset](https://drive.google.com/file/d/1dIyckJzUM5dpL7Xf6nr90m9j_Nk27HZZ/view?usp=sharing)
   - Extract and place the downloaded files in `RL_Index/train_data/`

2. **Format the Data**
   ```bash
   cd RL_Index/data_preprocess
   python build_dataset.py
   ```
   
   **Output:**
   - Formatted training dataset in `train_data/built_dataset/`
   - Training file: `v2_train/train.parquet`
   - Processing time: ~10-30 minutes depending on data size

---

## Evaluation Data Preparation

### Overview
Download and setup the BRIGHT benchmark for evaluation. BRIGHT is a comprehensive evaluation dataset across 8 different domains.

### Steps

```bash
cd RL_Index/data_preprocess
python get_eval_dataset.py
```

**Output:**
- BRIGHT evaluation datasets in `eval_data/BRIGHT/`
- Domains: "biology" "earth_science" "economics" "psychology" "sustainable_living" "robotics" "stackoverflow" "pony" "aops" "theoremqa_questions" "theoremqa_theorems" "leetcode"
- Total size: ~472M

### Dataset Statistics

| Metric | Biology | Earth Science | Economics | Psychology | Robotics | Stack Overflow | Sustainable Living | LeetCode | Pony | AoPS | TheoremQA-Q | TheoremQA-T |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Queries ($Q$)** | 103 | 116 | 103 | 101 | 101 | 117 | 108 | 142 | 112 | 111 | 194 | 76 |
| **Documents ($D$)** | 57,359 | 121,249 | 50,220 | 52,835 | 61,961 | 107,081 | 60,792 | 413,932 | 7,894 | 188,002 | 188,002 | 23,839 |

---

## Model Training

### Overview
Train the Llama 3.2-3B-Instruct model using the GRPO (Group Relative Policy Optimization) algorithm for RL-based index reasoning.

### Prerequisites
- Two separate compute nodes with GPU support
- Each node with 24GB+ VRAM
- Network connectivity between nodes

### Setup & Execution

#### Step 1: Node 1 - Start vLLM Inference Server

```bash
cd RL_Index/scripts/train_llama
bash run_server.sh
```

**Configuration options in `run_server.sh`:**
- `model_name`: Base model to load
- `tensor_parallel_size`: Number of GPUs for parallelism
- `port`: Server port (default: 8000)
- `gpu_memory_utilization`: GPU memory allocation ratio

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

#### Step 2: Node 2 - Run GRPO Training

```bash
cd RL_Index/scripts/train_llama
bash run_train.sh
```

**Key training parameters in `run_train.sh`:**
- `learning_rate`: Learning rate for optimization (recommend: 5e-6)
- `batch_size`: Training batch size (recommend: 32)
- `num_train_epochs`: Number of training epochs
- `save_steps`: Save checkpoint every N steps
- `vllm_server_url`: vLLM server address

**Expected training time:** 
- 100 epochs on single GPU: ~24-48 hours
- Training will save checkpoints at specified intervals

### Monitoring Training

```bash
# Check training logs
tail -f outputs/train_logs.txt

# Monitor GPU usage (on training node)
nvidia-smi -l 1
```

### Saving and Loading Checkpoints

Checkpoints are automatically saved to `outputs/checkpoints/` during training. To resume training:

```bash
bash run_train.sh --resume_from_checkpoint outputs/checkpoints/checkpoint-1000
```

---

## Document Reasoning & Augmentation

### Overview
Generate augmented documents using the trained RL model for improved retrieval. This step creates reasoning-enhanced versions of documents that provide better context for retrieval.

### Prerequisites
- Completed model training or available trained checkpoint
- Original evaluation documents

### Execution

```bash
cd RL_Index/scripts/gen_and_indexing
bash run_doc_rewriting.sh
```

**Configuration in `run_doc_rewriting.sh`:**
- `ckpt_path`: Path to trained model checkpoint
- `data_dir`: Directory containing original documents
- `output_dir`: Where to save augmented documents
- `batch_size`: Inference batch size

**Output:**
- Augmented documents with reasoning insights
- Format: `.parquet` files in organized dataset structure
- Expected time: 2-6 hours depending on dataset size

### Document Augmentation Details

The augmentation process adds:
- **Key points:** Important information extracted from documents
- **Explanations:** Reasoning and context for better understanding
- **Main topics:** Categorization and topic labeling

---

## Embedding and Indexing

### Overview
Create embeddings using pre-trained models and build FAISS indices for efficient similarity-based retrieval.

### Supported Embedding Models

| Model | Context Length | Dimension | Speed | Best Use Case |
|-------|-----------------|-----------|-------|---------------|
| `BAAI/bge-large-en-v1.5` | 512 | 1024 | Fast | General retrieval |
| `sentence-transformers/all-mpnet-base-v2` | 384 | 768 | Very Fast | Resource-limited |
| `Alibaba-NLP/gte-Qwen1.5-7B-instruct` | 16384 | 4096 | Slower | Long documents |
| `intfloat/e5-mistral-7b-instruct` | 32768 | 1024 | Slower | Very long documents |

### Index Types

- **Flat Index:** Exact brute-force search (recommended for datasets < 1M docs)
- **HNSW:** Approximate nearest neighbor search (recommended for large datasets)

### Execution

#### Option 1: Embed Baseline (Original) Documents

```bash
cd RL_Index/scripts/gen_and_indexing/baseline
python emb_and_index.py \
  --model "BAAI/bge-large-en-v1.5" \
  --dataset "pony" \
  --benchmark "bright" \
  --device "0" \
  --index_type "flat"
```

#### Option 2: Embed Augmented Documents

```bash
cd RL_Index/scripts/gen_and_indexing
python emb_and_index.py \
  --model "BAAI/bge-large-en-v1.5" \
  --dataset "pony" \
  --benchmark "bright" \
  --device "0" \
  --index_type "flat" \
  --step 1000 \
  --version "La_SBERT_RL_1000"
```

### Parameter Details

- `--model`: Embedding model name (default: `BAAI/bge-large-en-v1.5`)
- `--dataset`: Dataset to index (`pony`, `aops`, `leetcode`, `biology`, etc.)
- `--benchmark`: Benchmark name (default: `bright`)
- `--device`: GPU device ID (e.g., `0`, `1`)
- `--index_type`: `flat` for exact search or `hnsw` for approximate
- `--step`: RL training step for augmented documents
- `--version`: Version identifier for tracking different augmentations
- `--components`: Components to use (`I`, `O`, `M`, `K`, `E` or combinations like `M+K+E`)
- `--is_multi_content`: Whether to use multiple content versions

### Output Structure

```
embeddings/
├── bright/
│   └── pony/
│       ├── baseline/
│       │   ├── bge-large-en-v1.5_flat_index.faiss
│       │   └── index_id_dict.pkl
│       └── La_SBERT_RL_1000/
│           ├── bge-large-en-v1.5_flat_index.faiss
│           └── index_id_dict.pkl
```

---

## Evaluation

### Overview
Evaluate retrieval performance on the BRIGHT benchmark using the created indices. This measures how well the augmented documents improve retrieval quality.

### Prerequisites
- Embedded baseline documents
- Embedded augmented documents
- Evaluation queries from BRIGHT benchmark

### Execution

#### For Dense Retrieval Models (BGE, MPNET)

```bash
cd RL_Index/scripts/eval
bash eval.sh
```

#### For Large Language Model Embeddings (GTE-Qwen, E5)

```bash
cd RL_Index/scripts/eval
bash eval_LM.sh
```

### Evaluation Metrics

| Metric | Description | Range | Better When |
|--------|-------------|-------|-------------|
| **NDCG@10** | Normalized Discounted Cumulative Gain | 0-1 | Closer to 1 |
| **Recall@1** | Fraction of queries with relevant doc in top 1 | 0-1 | Higher |
| **Recall@5** | Fraction of queries with relevant doc in top 5 | 0-1 | Higher |
| **Recall@10** | Fraction of queries with relevant doc in top 10 | 0-1 | Higher |
| **Recall@20** | Fraction of queries with relevant doc in top 20 | 0-1 | Higher |

### Output Files

- `results/`: Retrieved documents for analysis
- `metrics/`: Computed evaluation metrics
- `logs/`: Evaluation logs and debug information

---

## Project Structure

```
RL_Index/
├── data_preprocess/
│   ├── build_dataset.py           # Training data formatting
│   ├── get_eval_dataset.py        # Evaluation data downloading
│   ├── train_data/                # Training dataset storage
│   └── eval_data/                 # Evaluation dataset storage
├── scripts/
│   ├── train_llama/
│   │   ├── run_server.sh          # vLLM server startup
│   │   └── run_train.sh           # GRPO training script
│   ├── gen_and_indexing/
│   │   ├── baseline/              # Baseline embedding scripts
│   │   └── run_doc_rewriting.sh   # Document augmentation
│   │   └── run_emb_and_indx.sh   # Embedding and Indexing
│   └── eval/
│       ├── eval.sh                # Evaluation for dense models
│       └── eval_LM.sh             # Evaluation for LM models
├── outputs/                       # Training outputs and checkpoints
├── results/                       # Evaluation results
└── README.md                      # This file
```


---

## Performance Benchmarks

### Results on BRIGHT Benchmark

TO BE ADDED
---

## Citation

If you use RL-Index in your research, please cite:

TO BE ADDED

---

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

## Acknowledgments

- Built on top of [TongSearch-QR](https://github.com/tongsearch/tongsearch-qr)
- Embedding models from [HuggingFace](https://huggingface.co)
- Retrieval framework using [FAISS](https://github.com/facebookresearch/faiss)
- Training framework: [vLLM](https://github.com/lm-sys/vllm)

---

**Last Updated:** April 2026 | **Version:** 1.0.0
