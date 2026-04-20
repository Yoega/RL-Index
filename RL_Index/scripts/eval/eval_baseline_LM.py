import pandas as pd
import numpy as np
import faiss
from typing import Dict, List
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import AutoTokenizer
from vllm import LLM
import os
import logging
import pickle


MODEL_MAX_LEN_DICT = {
    "intfloat/e5-mistral-7b-instruct": 8192,
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


def evaluate_dataset(dataset_name: str, 
                    index_path: str,
                    model: LLM,
                    query_path: str,
                    qrel_path: str,
                    index_id_dict_path: str,
                    args,
                    k: int = 10,
                    tokenizer: AutoTokenizer = None) -> Dict[str, float]:
    """Evaluate NDCG@k for a dataset using Faiss index"""

    qid_list = []
    top_k_docid_list = []

    # Load Faiss index
    index = faiss.read_index(index_path)

    # Load query 
    logging.info(f"Loading query")
    query_df = pd.read_parquet(query_path)
    print(f"query_df shape: {query_df.shape}")
    print(query_df.head())

    # Load index_id_dict
    with open(index_id_dict_path, "rb") as f:
        index_id_dict = pickle.load(f)

    # extract query embedding
    query_embeddings = []

    logging.info(f"Extracting query embeddings")
    for _, row in tqdm(query_df.iterrows()):            
        query = row['query']
        tokenized_query = tokenizer.tokenize(query)
        if len(tokenized_query) > MODEL_MAX_LEN_DICT[args.model]:
            tokenized_query = tokenized_query[:MODEL_MAX_LEN_DICT[args.model]]
            query = tokenizer.convert_tokens_to_string(tokenized_query)

        query_embedding = model.encode(query, use_tqdm=False)
        query_embeddings.append(query_embedding[0].outputs.embedding)

    query_embeddings = np.array(query_embeddings, dtype=np.float32)
    
    logging.info(f"query embeddings shape: {query_embeddings.shape}")
    
    # Load qrels
    qrel_df = pd.read_parquet(qrel_path)
    print(f"qrel_df shape: {qrel_df.shape}")
    print(qrel_df.head())
    
    logging.info(f"qrel df loaded")
    
    # Convert qrels to dictionary format {query_id: {doc_id: relevance}}
    qrels_dict = {}
    for _, row in qrel_df.iterrows():
        if row['query-id'] not in qrels_dict:
            qrels_dict[row['query-id']] = {}
        qrels_dict[row['query-id']][str(row['corpus-id'])] = row['score']
    
    # Calculate NDCG@k for each query
    ndcg_scores = {}
    
    for i, query_embedding in tqdm(enumerate(query_embeddings)):
        # Search using Faiss
        normalize_query_embedding = query_embedding.reshape(1, -1)
        
        # normalize query embedding for cosine similarity
        faiss.normalize_L2(normalize_query_embedding)

        excluded_ids = query_df.loc[i, "excluded_ids"]

        first_k = k * 200
        scores, I = index.search(normalize_query_embedding, first_k)

        query_id = query_df.loc[i, "query-id"]
        
        # convert I to corpus-id
        corpus_ids_list = [str(index_id_dict[idx]) for idx in I[0]]

        # Create lists of (score, corpus_id) tuples
        corpus_score_id = list(zip(scores[0], corpus_ids_list))
        
        # Create a dictionary for scores for easy lookup
        scores_dict = {corpus_id: score for score, corpus_id in corpus_score_id}
        
        # Combine all scores and create the final list
        final_scores = [(score, corpus_id) for corpus_id, score in scores_dict.items()]

        if len(final_scores) < k:
            raise ValueError(f"len(final_scores) < k: {len(final_scores)} < {k}")
        
        # Sort by score in descending order
        final_scores.sort(reverse=True)
        
        # Remove excluded IDs
        final_scores = [score_id for score_id in final_scores if score_id[1] not in excluded_ids]

        if len(final_scores) < k:
            raise ValueError(f"len(final_scores) < k: {len(final_scores)} < {k}")
        
        # Truncate to k
        final_scores = final_scores[:k]

        corpus_ids = [score_id[1] for score_id in final_scores]

        qid_list.append(query_id)
        top_k_docid_list.append(corpus_ids)

        # Calculate NDCG@k
        if str(query_id) in qrels_dict:
            ndcg = calculate_ndcg(corpus_ids, qrels_dict[str(query_id)], k)
            ndcg_scores[str(query_id)] = ndcg
    
    # Calculate mean NDCG@k
    mean_ndcg = np.mean(list(ndcg_scores.values()))
    
    return {
        'ndcg_scores': ndcg_scores,
        'mean_ndcg': mean_ndcg,
        'qid_list': qid_list,
        'top_k_docid_list': top_k_docid_list,
    }

if __name__ == "__main__":
    args = get_config()
    logging.basicConfig(level=logging.INFO)
    model_name = args.model.split("/")[-1]

    result_dir = f"./result/{args.benchmark}/{args.dataset}/{model_name}/{args.query_type}"
    os.makedirs(result_dir, exist_ok=True)

    logging.info(f"Loading model")
    model = LLM(
        model=args.model,
        device="cuda:0",
        seed=42,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    benchmark = args.benchmark
    dataset_name = args.dataset
    
    capitalized_benchmark_name = benchmark.upper()

    data_dir = "embeddings"
    index_path = f"{data_dir}/{benchmark}/{dataset_name}/aug/LM_{model_name}_{args.index_type}_index.faiss"
    
    if args.query_type == "original_query":
        query_path = f"data/{capitalized_benchmark_name}/{dataset_name}/query.parquet"
    else:
        query_path = f"data/{capitalized_benchmark_name}/{dataset_name}/{args.query_type}_query.parquet"

    qrel_path = f"data/{capitalized_benchmark_name}/{dataset_name}/qrel.parquet"
    index_id_dict_path = f"{data_dir}/{benchmark}/{dataset_name}/aug/index_id_dict.pkl"

    results = evaluate_dataset(
        dataset_name=dataset_name,
        index_path=index_path,
        model=model,
        query_path=query_path,
        qrel_path=qrel_path,
        index_id_dict_path=index_id_dict_path,
        args=args,
        k=args.k,
        tokenizer=tokenizer
    )

    # save ndcg scores with text
    mean_ndcg = results['mean_ndcg']
    with open(f"{result_dir}/ndcg{args.k}_scores.txt", "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Benchmark: {benchmark}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Query Type: {args.query_type}\n")
        f.write(f"Index Type: {args.index_type}\n")
        f.write("\n")
        f.write(f"Mean NDCG@{args.k} for {dataset_name}: {mean_ndcg:.4f}\n")

    # save result
    qid_list = results['qid_list']
    top_k_docid_list = results['top_k_docid_list']

    analysis_df = pd.DataFrame({
        'query-id': qid_list,
        'top_k_docid': top_k_docid_list,    
    })

    print(analysis_df.head())
    analysis_df.to_parquet(f"{result_dir}/analysis_df.parquet", index=False)
    
    logging.info(f"Mean NDCG@{args.k} for {dataset_name}: {results['mean_ndcg']*100:.1f}")
