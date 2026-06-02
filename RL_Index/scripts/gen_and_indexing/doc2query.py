"""
Document rewriting script for augmenting documents and queries using Doc2Query.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def generate_augmented_documents(model, tokenizer, eval_df, output_path, batch_size=16, num_queries=3):
    """
    Generate augmented documents using the Doc2Query model.

    Args:
        model: Hugging Face model instance for inference
        tokenizer: Hugging Face tokenizer instance
        eval_df: DataFrame containing documents with 'id' and 'content' columns
        output_path: Path to save the output parquet file
        batch_size: Number of documents to process in parallel
        num_queries: Number of queries to generate per document
    """
    device = model.device
    doc_ids = []
    augmented_texts = []

    # Get total number of documents
    total_docs = len(eval_df)
    
    # Process in batches
    for i in tqdm(range(0, total_docs, batch_size), desc="Generating queries"):
        batch = eval_df.iloc[i:i + batch_size]
        
        batch_doc_ids = batch['id'].tolist()
        batch_contents = batch['content'].tolist()

        # Tokenize input
        inputs = tokenizer(
            batch_contents,
            padding=True,
            truncation=True,
            max_length=500,
            return_tensors="pt"
        ).to(device)

        # Generate queries
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=64,
                num_return_sequences=num_queries,
                do_sample=True,
                top_k=10,
                temperature=1.0
            )

        # Decode generated queries
        decoded_queries = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Group queries by document and concatenate them
        for j in range(len(batch_contents)):
            doc_id = batch_doc_ids[j]
            # Extract the num_queries generated for the j-th document
            start_idx = j * num_queries
            end_idx = start_idx + num_queries
            doc_queries = decoded_queries[start_idx:end_idx]
            
            # Combine the generated queries into a single string
            combined_queries = " ".join(doc_queries)
            
            # append combined queries to the original document content
            combined_queries = batch_contents[j] + " " + combined_queries
            
            doc_ids.append(doc_id)
            augmented_texts.append(combined_queries)

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
    parser = argparse.ArgumentParser(description="Augment documents or queries using Doc2Query model.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path or HuggingFace ID for the Doc2Query model (e.g., macavaney/doc2query-t5-base-msmarco)")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to the input dataset directory")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset to load from the input directory")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save augmented documents/queries")
    parser.add_argument("--rewritten_content", type=str, default="document", choices=["document", "query"],
                        help="Type of content to augment (document or query)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for model inference")
    parser.add_argument("--num_queries", type=int, default=10,
                        help="Number of queries to generate per document")
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Tokenizer and Model
    print(f"Loading model {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)
    model.eval()

    # Process based on content type
    if args.rewritten_content == "document":
        input_parquet = Path(args.input_path) / f"{args.dataset_name}/document.parquet"
        print(f"Reading dataset from {input_parquet}...")
        eval_df = pd.read_parquet(input_parquet)
        
        generate_augmented_documents(
            model=model, 
            tokenizer=tokenizer, 
            eval_df=eval_df, 
            output_path=args.output_path,
            batch_size=args.batch_size,
            num_queries=args.num_queries
        )
    elif args.rewritten_content == "query":
        raise NotImplementedError("Query augmentation is not yet implemented")


if __name__ == "__main__":
    main()