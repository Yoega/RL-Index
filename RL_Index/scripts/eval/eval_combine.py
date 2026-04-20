import pandas as pd
import numpy as np
import faiss
from typing import Dict, List
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import os
import logging
import pickle

MODEL_MAX_LEN_DICT = {
    "intfloat/e5-mistral-7b-instruct": 8192,
    "Salesforce/SFR-Embedding-Mistral": 4090,
    "GritLM/GritLM-7B": 8192,
    "Alibaba-NLP/gte-Qwen1.5-7B-instruct": 16384,
}


def get_config():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="Alibaba-NLP/gte-Qwen1.5-7B-instruct")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--benchmark", type=str, default="bright")
    parser.add_argument("--dataset", type=str, default="pony")
    parser.add_argument("--id_col_name", type=str, default="id")
    parser.add_argument("--query_type", type=str, default="original_query")
    parser.add_argument("--index_type", type=str, default="flat")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--analysis_mode", type=bool, default=False)
    args = parser.parse_args()
    return args

    
def calculate_dcg(relevance_scores: List[float], k: int) -> float:
    """Calculate DCG@k"""
    dcg = 0
    for i in range(min(len(relevance_scores), k)):
        dcg += relevance_scores[i] / np.log2(i + 2)
    return dcg

def calculate_ndcg(predicted_ranks: List[int], true_relevance: Dict[str, int], k: int) -> float:
    """Calculate NDCG@k for a single query"""
    # Get relevance scores for predicted ranks
    pred_relevance = [true_relevance.get(str(rank), 0) for rank in predicted_ranks[:k]]
    
    # Calculate ideal DCG
    ideal_relevance = sorted(true_relevance.values(), reverse=True)[:k]
    
    dcg = calculate_dcg(pred_relevance, k)
    idcg = calculate_dcg(ideal_relevance, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

# compute recall
def calculate_recall(predicted_ranks: List[int], true_relevance: Dict[str, int], k: int) -> float:
    """Calculate Recall@k for a single query"""
    relevant_items = set(true_relevance.keys())
    retrieved_items = set(str(rank) for rank in predicted_ranks[:k])
    
    true_positives = len(relevant_items.intersection(retrieved_items))
    total_relevant = len(relevant_items)
    
    if total_relevant == 0:
        return 0.0
    
    return true_positives / total_relevant

def evaluate_dataset(dataset_name: str, 
                    original_index_path: str,
                    aug_index_path: str,
                    model: SentenceTransformer,
                    query_path: str,
                    qrel_path: str,
                    original_index_id_dict_path: str,
                    aug_index_id_dict_path: str,
                    args,
                    k: int = 10,
                    tokenizer: AutoTokenizer = None) -> Dict[str, float]:
    """Evaluate NDCG@k and Recall@k by combining results from two indices"""

    qid_list = []
    top_k_docid_list = []

    # Load Faiss indices
    original_index = faiss.read_index(original_index_path)
    aug_index = faiss.read_index(aug_index_path)

    # Load data
    logging.info(f"Loading query")
    query_df = pd.read_parquet(query_path)
    print(f"query_df shape: {query_df.shape}")
    print(query_df.head())

    with open(original_index_id_dict_path, "rb") as f:
        original_index_id_dict = pickle.load(f)

    with open(aug_index_id_dict_path, "rb") as f:
        aug_index_id_dict = pickle.load(f)
        print(f"aug_index_id_dict loaded, length: {len(aug_index_id_dict)}")

    # Extract query embeddings
    query_embeddings = []
    logging.info(f"Extracting query embeddings")
    for _, row in tqdm(query_df.iterrows()):           
        query = row['query']
        query_embedding = model.encode(query, show_progress_bar=False)
        query_embeddings.append(query_embedding)

    query_embeddings = np.array(query_embeddings, dtype=np.float32)
    logging.info(f"query embeddings shape: {query_embeddings.shape}")
    
    # Load qrels
    qrel_df = pd.read_parquet(qrel_path)
    print(f"qrel_df shape: {qrel_df.shape}")
    print(qrel_df.head())
    logging.info(f"qrel df loaded")
    
    # Convert qrels to dictionary
    qrels_dict = {}
    for _, row in qrel_df.iterrows():
        if row['query-id'] not in qrels_dict:
            qrels_dict[row['query-id']] = {}
        qrels_dict[row['query-id']][str(row['corpus-id'])] = row['score']
    
    # Calculate scores for each query
    ndcg_scores = {}
    recall_scores = {}
    
    for i, query_embedding in tqdm(enumerate(query_embeddings)):
        # Normalize and search both indices
        normalize_query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(normalize_query_embedding)

        excluded_ids = query_df.loc[i, "excluded_ids"]
        first_k = k * 100  # in case some ids are excluded

        # Search original and augmented indices
        ori_scores, ori_indices = original_index.search(normalize_query_embedding, first_k)
        aug_scores, aug_indices = aug_index.search(normalize_query_embedding, first_k)

        query_id = query_df.loc[i, "query-id"]

        # Convert indices to corpus IDs
        ori_corpus_ids = [str(original_index_id_dict[idx]) for idx in ori_indices[0]]
        aug_corpus_ids = [str(aug_index_id_dict[idx]) for idx in aug_indices[0]]

        # Combine scores: merge results and sum scores for overlapping documents
        combined_scores = {}
        for score, corpus_id in zip(ori_scores[0], ori_corpus_ids):
            combined_scores[corpus_id] = score
        
        for score, corpus_id in zip(aug_scores[0], aug_corpus_ids):
            if corpus_id in combined_scores:
                combined_scores[corpus_id] += score  # Sum scores for overlapping docs
            else:
                combined_scores[corpus_id] = score

        # Sort and filter
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        corpus_ids = [corpus_id for corpus_id, _ in sorted_results if corpus_id not in excluded_ids]

        if len(corpus_ids) < k:
            raise ValueError(f"len(corpus_ids) < k: {len(corpus_ids)} < {k}")
        
        corpus_ids = corpus_ids[:k]

        # Calculate metrics
        if str(query_id) in qrels_dict:
            ndcg = calculate_ndcg(corpus_ids, qrels_dict[str(query_id)], k)
            ndcg_scores[str(query_id)] = ndcg
            
            recall = calculate_recall(corpus_ids, qrels_dict[str(query_id)], k)
            recall_scores[str(query_id)] = recall
        
        if args.analysis_mode:
            qid_list.append(query_id)
            top_k_docid_list.append(corpus_ids)
    
    # Calculate mean scores
    mean_ndcg = np.mean(list(ndcg_scores.values()))
    mean_recall = np.mean(list(recall_scores.values()))
    
    result = {
        'ndcg_scores': ndcg_scores,
        'recall_scores': recall_scores,
        'mean_ndcg': mean_ndcg,
        'mean_recall': mean_recall,
    }
    
    if args.analysis_mode:
        result['qid_list'] = qid_list
        result['top_k_docid_list'] = top_k_docid_list
    
    return result


if __name__ == "__main__":
    args = get_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    logging.basicConfig(level=logging.INFO)
    model_name = args.model.split("/")[-1]

    result_dir = f"./result/{args.benchmark}/{args.dataset}/{model_name}/{args.query_type}"
    os.makedirs(result_dir, exist_ok=True)

    logging.info(f"Loading model")
    model = SentenceTransformer(args.model, device=f"cuda:0", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    benchmark = args.benchmark
    dataset_name = args.dataset
    capitalized_benchmark_name = benchmark.upper()

    data_dir = "embeddings"
    # Original and augmented index paths
    original_index_path = f"{data_dir}/{benchmark}/{dataset_name}/original_document/LM_{model_name}_{args.index_type}_index.faiss"
    aug_index_path = f"{data_dir}/{benchmark}/{dataset_name}/aug/LM_{model_name}_{args.index_type}_index.faiss"
    
    # Query and qrels paths
    if args.query_type == "original_query":
        query_path = f"data/{capitalized_benchmark_name}/{dataset_name}/query.parquet"
    else:
        query_path = f"data/{capitalized_benchmark_name}/{dataset_name}/{args.query_type}_query.parquet"

    qrel_path = f"data/{capitalized_benchmark_name}/{dataset_name}/qrel.parquet"
    
    # Index ID dict paths
    original_index_id_dict_path = f"{data_dir}/{benchmark}/{dataset_name}/original_document/index_id_dict.pkl"
    aug_index_id_dict_path = f"{data_dir}/{benchmark}/{dataset_name}/aug/index_id_dict.pkl"

    results = evaluate_dataset(
        dataset_name=dataset_name,
        original_index_path=original_index_path,
        aug_index_path=aug_index_path,
        model=model,
        query_path=query_path,
        qrel_path=qrel_path,
        original_index_id_dict_path=original_index_id_dict_path,
        aug_index_id_dict_path=aug_index_id_dict_path,
        args=args,
        k=args.k,
        tokenizer=tokenizer
    )

    # Save results
    mean_ndcg = results['mean_ndcg']
    mean_recall = results['mean_recall']
    
    with open(f"{result_dir}/combined_scores.txt", "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Benchmark: {benchmark}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Query Type: {args.query_type}\n")
        f.write(f"Index Type: {args.index_type}\n")
        f.write("\n")
        f.write(f"Mean Recall@{args.k}: {mean_recall:.4f}\n")
        f.write(f"Mean NDCG@{args.k}: {mean_ndcg:.4f}\n")

    # Save analysis results if in analysis mode
    if args.analysis_mode:
        analysis_df = pd.DataFrame({
            'query-id': results['qid_list'],
            'top_k_docid': results['top_k_docid_list'],    
        })
        analysis_df.to_parquet(f"{result_dir}/analysis_df.parquet", index=False)
    
    print(f"Mean NDCG@{args.k}: {results['mean_ndcg']*100:.1f}%")
    print(f"Mean Recall@{args.k}: {results['mean_recall']*100:.1f}%")
    print(f"Result dir: {result_dir}")