# RL-Index: Reinforcement Learning for Retrieval Index Reasoning

An end-to-end pipeline for enhancing retrieval by RL-based index reasoning.

## Table of Contents
1. [Training Data Preparation](#training-data-preparation)
2. [Evaluation Data Preparation](#evaluation-data-preparation)
3. [Model Training](#model-training)
4. [Documents Reasoning / Augmentation](#documents-augmentation)
5. [Embedding and Indexing](#embedding-and-indexing)
6. [Evaluation](#evaluation)

---

## Training Data Preparation
*Detailed instructions on how to clean and format raw data for model training.*

- **Data Sources:** Where the raw data is pulled from (e.g., S3 buckets, local SQL databases).
- **Preprocessing Scripts:** - `clean_data.py`: Removes noise, duplicates, and handles missing values.
    - `format_labels.py`: Converts raw annotations into the required format (e.g., JSONL, CSV).
- **Usage:**
  ```bash
  python src/prep_training.py --input ./raw_data --output ./proc_data
