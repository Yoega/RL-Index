#!/usr/bin/env python3
"""
This code is sourced from TongSearchQR.
Reference: https://github.com/bigai-nlco/TongSearch-QR
"""

import argparse
import json
import logging
import random
from pathlib import Path
import pandas as pd

from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process sampled question-answer JSON and save as HuggingFace dataset."
    )
    parser.add_argument("--input", type=str, default="train_data/v2_train.parquet", 
                        help="Path to input parquet file") 
    parser.add_argument("--output_dir", type=str, default="train_data/built_dataset",
                        help="Directory to save the output parquet")
    parser.add_argument("--output_name", type=str, default="v2_train",
                        help="Name for output file")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    parser.add_argument("--max_length", type=int, default=7000, help="Maximum allowed question length")
    return parser.parse_args()


def build_system_prompt() -> str:
    """Return system prompt. (can be modified based on different tasks)"""
    return ""
    
def build_user_prompt(doc: str) -> str:
    """Build user prompt given the document content."""
    document_prompt = f"""
You are an advanced language model specializing in knowledge extraction and user need modeling. \
Your task is to extract hypothetical user scenarios from a given document, \
ensuring that the generated information needs reflect the document's overall insights and knowledge, \
rather than isolated details.

Content:
- Key Points: Summarize the core concepts, insights, or knowledge presented
- Explanation: Explain how the document fulfills hypothetical user need, \
ensuring that explanations are generalized and conceptual rather than overly detailed.

<Document>
{doc}
    """

    return document_prompt

def process_data(data_df: pd.DataFrame, max_length: int) -> tuple[list, int]:
    """
    Process sampled question-document pairs into dataset items.
    Returns:
        (list of items, max question length)
    """
    system_prompt = build_system_prompt()
    processed_data = []
    max_len = 0

    for _, row in data_df.iterrows():
        query = row['query']
        pos = row['pos']

        # apply prompt wrapper
        pos_prompt = build_user_prompt(pos)

        # truncate long input
        if len(tokenizer(pos_prompt)['input_ids']) > max_length:
            pos_prompt = tokenizer.decode(tokenizer(pos_prompt)['input_ids'][:max_length])
            print(f"Truncated long input document.\n{pos_prompt}")

        # Messages for chat-style dataset
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pos_prompt}
        ]

        # Track max length
        max_len = max(max_len, len(pos_prompt))

        # Store solution metadata as JSON string
        solution = {
            "query": query,
            "pos": pos
        }

        item_dict = {
            "prompt": messages,
            "answer": json.dumps(solution, ensure_ascii=False)
        }
        processed_data.append(item_dict)

    return processed_data, max_len


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Load input data
    logging.info("Loading input file: %s", args.input)
    data_df = pd.read_parquet(args.input)
    # Process data
    logging.info("Processing data...")
    dataset_items, max_len = process_data(data_df, max_length=args.max_length)
    random.seed(args.seed)
    random.shuffle(dataset_items)
    dataset = Dataset.from_list(dataset_items)

    # Save output
    output_path = Path(args.output_dir) / f"{args.output_name}/train.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(output_path))

    logging.info("Dataset saved: %s", output_path)
    logging.info("Total samples: %d | Max question length: %d", len(dataset_items), max_len)


if __name__ == "__main__":
    main()
