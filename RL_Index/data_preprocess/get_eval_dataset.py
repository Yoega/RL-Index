import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Setup and Data Loading
cache_dir = "./"
save_root = "eval_data/BRIGHT"

print("Loading datasets...")
ds_docs = load_dataset("xlangai/BRIGHT", "documents", cache_dir=cache_dir)
ds_examples = load_dataset("xlangai/BRIGHT", "examples", cache_dir=cache_dir)
ds_gpt4 = load_dataset("xlangai/BRIGHT", "gpt4_reason", cache_dir=cache_dir)

dataset_list = list(ds_examples.keys())

# Processing Loop
for dataset in tqdm(dataset_list, desc="Processing Datasets"):
    # Create directory for the specific subset
    dataset_path = os.path.join(save_root, dataset)
    os.makedirs(dataset_path, exist_ok=True)
    
    # Corpus
    if dataset in ds_docs:
        df_docs = ds_docs[dataset].to_pandas()
        df_docs.to_parquet(os.path.join(dataset_path, "document.parquet"))

    # Queries
    query_df = ds_examples[dataset].to_pandas()
    gpt4_df = ds_gpt4[dataset].to_pandas()
    
    # Selection and Renaming
    cols = ["query", "id", "excluded_ids"]
    query_df = query_df[cols].rename(columns={"id": "query-id"})
    gpt4_df = gpt4_df[cols].rename(columns={"id": "query-id"})
    
    # Clean excluded_ids (convert to list and handle "N/A")
    def clean_excluded(ids):
        ids_list = list(ids)
        return [] if ids_list == ["N/A"] else ids_list

    query_df["excluded_ids"] = query_df["excluded_ids"].apply(clean_excluded)
    gpt4_df["excluded_ids"] = gpt4_df["excluded_ids"].apply(clean_excluded)
    
    query_df.to_parquet(os.path.join(dataset_path, "query.parquet"))
    gpt4_df.to_parquet(os.path.join(dataset_path, "gpt4_reason_query.parquet"))

    # Qrels
    # We use the full original examples dataframe to get gold_ids
    full_query_data = ds_examples[dataset].to_pandas()
    
    qrel_records = []
    for _, row in full_query_data.iterrows():
        qid = row["id"]
        q_text = row["query"]
        gold_ids = row["gold_ids"]
        
        for gold_id in gold_ids:
            qrel_records.append({
                "query": q_text,
                "query-id": qid,
                "corpus-id": gold_id,
                "score": 1
            })
            
    df_qrel = pd.DataFrame(qrel_records)
    df_qrel.to_parquet(os.path.join(dataset_path, "qrel.parquet"))

print(f"Done! All files saved to {save_root}")