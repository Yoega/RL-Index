# RL-Index: Reinforcement Learning for Retrieval Index Reasoning

An end-to-end pipeline for enhancing retrieval through reinforcement learning-based index reasoning.

## Overview

RL-Index uses RL techniques to improve document retrieval by training models to perform intelligent index reasoning. This repository contains the complete pipeline from data preparation through evaluation.

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Training Data Preparation](#training-data-preparation)
4. [Evaluation Data Preparation](#evaluation-data-preparation)
5. [Model Training](#model-training)
6. [Document Reasoning & Augmentation](#document-reasoning--augmentation)
7. [Embedding and Indexing](#embedding-and-indexing)
8. [Evaluation](#evaluation)

---

## Requirements

- Python 3.10+
- CUDA 11.8+
- GPU with 24GB+ VRAM (for model training)

## Installation

```bash
# Clone the repository
git clone https://github.com/Yoega/RL-Index.git
cd RL-Index

# Create and activate virtual environment (optional)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Training Data Preparation

### Overview
Prepare training data from the TongSearch-QR benchmark for model fine-tuning.

### Steps

1. **Download Data**
   - Download the training dataset from: [TongSearch-QR V2 Dataset](https://drive.google.com/file/d/1dIyckJzUM5dpL7Xf6nr90m9j_Nk27HZZ/view?usp=sharing)
   - Extract and place it in `RL_Index/train_data/`

2. **Format the Data**
   ```bash
   cd RL_Index/data_preprocess
   python build_dataset.py
   ```
   This will process and format the raw data for model training.

---

## Evaluation Data Preparation

### Overview
Download and setup the BRIGHT benchmark for evaluation.

### Steps

```bash
cd RL_Index/data_preprocess
python get_eval_dataset.py
```

This script will automatically download and organize the BRIGHT evaluation dataset.

---

## Model Training

### Overview
Train the Llama 3.2-3B-Instruct model using the GRPO algorithm.

### Setup & Execution

The training requires two nodes:
1. **Node 1:** Run the vLLM server
2. **Node 2:** Run the GRPO training algorithm

#### Node 1 - Start vLLM Server:
```bash
cd RL_Index/scripts/train_llama
bash run_server.sh
```

#### Node 2 - Run GRPO Training:
```bash
cd RL_Index/scripts/train_llama
bash run_train.sh
```

### Configuration
Modify training parameters in `run_train.sh` as needed (learning rate, batch size, number of epochs, etc.).

---

## Document Reasoning & Augmentation

### Overview
Generate augmented documents using the trained model for improved retrieval.

### Execution

```bash
cd RL_Index/scripts/gen_and_indexing
bash run_doc_rewriting.sh
```

This generates reasoning-enhanced versions of documents in the evaluation dataset.

---

## Embedding and Indexing

### Overview
Create embeddings and build FAISS indices for efficient retrieval.

### Supported Models
- `Alibaba-NLP/gte-Qwen1.5-7B-instruct` (default)
- `intfloat/e5-mistral-7b-instruct`
- `BAAI/bge-large-en-v1.5`
- `sentence-transformers/all-mpnet-base-v2`

### Execution

```bash
cd RL_Index/scripts/gen_and_indexing
python emb_and_index.py \
  --model "Alibaba-NLP/gte-Qwen1.5-7B-instruct" \
  --dataset "pony" \
  --benchmark "bright" \
  --device "0"
```

**Key Parameters:**
- `--model`: Embedding model to use
- `--dataset`: Dataset name (e.g., pony, aops, leetcode)
- `--benchmark`: Benchmark name (default: bright)
- `--device`: GPU device ID
- `--index_type`: Index type - "flat" (default) or "hnsw"

---

## Evaluation

### Overview
Evaluate retrieval performance using the BRIGHT benchmark.

### Execution

```bash
cd RL_Index/scripts/eval
bash eval.sh
```

### Evaluation Metrics
- MRR (Mean Reciprocal Rank)
- NDCG (Normalized Discounted Cumulative Gain)
- Recall@K (K=1, 5, 10, 20)

---

## Project Structure

```
RL_Index/
├── data_preprocess/          # Data preparation scripts
├── scripts/
│   ├── train_llama/          # Model training scripts
│   ├── gen_and_indexing/     # Embedding and indexing scripts
│   ├── eval/                 # Evaluation scripts
│   └── baseline/             # Baseline methods
├── data/                     # Data storage directory
└── outputs/                  # Results and logs
```

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{lei2024rlindex,
  title={RL-Index: Reinforcement Learning for Retrieval Index Reasoning},
  year={2024}
}
```

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.
