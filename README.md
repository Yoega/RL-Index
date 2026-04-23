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
*Detailed instructions on how to get and format raw data for model training.*

- **Data Sources:** The training data used is from the paper "TongSearch-QR: Reinforced Query Reasoning for Retrieval". You can download it from [TODO: add the link]. Then, put the data under the directory [TODO: add the directory].
- **Format the Data**: run the following command to format the data for model training.
- **Preprocessing Scripts:** - `clean_data.py`: Removes noise, duplicates, and handles missing values.
    - `format_labels.py`: Converts raw annotations into the required format (e.g., JSONL, CSV).
- **Usage:**
  ```bash
  python src/prep_training.py --input ./raw_data --output ./proc_data


## Evaluation Data Preparation

## Model Training

## Document Reasoning / Augmentation

## Embedding and Indexing

## Evaluation

