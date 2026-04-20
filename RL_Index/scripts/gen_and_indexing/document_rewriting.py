"""
Document rewriting script for augmenting documents and queries using a language model.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from vllm_model import VllmModel


# Prompt for augmenting documents
DOCUMENT_PROMPT_TEMPLATE = """You are an advanced language model specializing in knowledge extraction and user need modeling. \
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


def build_document_prompt(doc: str) -> str:
    """Build the document prompt for augmentation."""
    return DOCUMENT_PROMPT_TEMPLATE.format(doc=doc)


def generate_augmented_documents(model, eval_df, output_path):
    """
    Generate augmented documents using the model.

    Args:
        model: VllmModel instance for inference
        eval_df: DataFrame containing documents with 'id' and 'content' columns
        output_path: Path to save the output parquet file
    """
    prompts = []
    doc_ids = []

    # Build prompts for all documents
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Building prompts"):
        doc_id = row['id']
        content = row['content']
        prompt = build_document_prompt(content)

        prompts.append(prompt)
        doc_ids.append(doc_id)

    # Generate augmentations in batch
    augmented_texts = model.predict_batch(prompts)

    # Create result dataframe
    result_df = pd.DataFrame({
        "id": doc_ids,
        "aug_content": augmented_texts
    })

    save_augmentations(result_df, output_path)


def save_augmentations(df, output_path):
    """
    Save augmented documents to a parquet file.

    Args:
        df: DataFrame to save
        output_path: Path to save the parquet file
    """
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    df.to_parquet(output_path)
    print(f"Augmented documents saved to {output_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Augment documents or queries using a language model.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the Vllm model checkpoint")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to the input dataset directory")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset to load from the input directory")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save augmented documents/queries")
    parser.add_argument("--rewritten_content", type=str, default="document", choices=["document", "query"],
                        help="Type of content to augment (document or query)")
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Initialize model with empty system prompt
    model = VllmModel(args.model_path, system_prompt="")

    # Process based on content type
    if args.rewritten_content == "document":
        input_parquet = Path(args.input_path) / f"{args.dataset_name}/document.parquet"
        eval_df = pd.read_parquet(input_parquet)
        generate_augmented_documents(model, eval_df, args.output_path)
    elif args.rewritten_content == "query":
        raise NotImplementedError("Query augmentation is not yet implemented")


if __name__ == "__main__":
    main()