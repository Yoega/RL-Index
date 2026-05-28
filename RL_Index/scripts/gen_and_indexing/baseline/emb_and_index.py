"""
This script is designed to generate embeddings for documents and build a FAISS index for efficient retrieval. 
It supports both flat and HNSW indexing methods. 
The supported models include "intfloat/e5-mistral-7b-instruct" and "Alibaba-NLP/gte-Qwen1.5-7B-instruct", which have different maximum token lengths.

"""

import faiss
import numpy as np
from vllm import LLM
from argparse import ArgumentParser
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import pickle
from tqdm import tqdm
import logging


MODEL_MAX_LEN_DICT = {
    "intfloat/e5-mistral-7b-instruct": 8192,
    "Alibaba-NLP/gte-Qwen1.5-7B-instruct": 16384,
}


def safe_convert_to_string(value):
    """Safely convert any value to string, handling lists and nested structures"""
    if isinstance(value, list):
        return '\n'.join(str(item) for item in value)
    elif isinstance(value, dict):
        return str(value)
    else:
        return str(value)



def get_config():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="Alibaba-NLP/gte-Qwen1.5-7B-instruct", help="The model to use for embedding generation")
    parser.add_argument("--input_file", type=str, default="", help="The path to the input file containing documents")
    parser.add_argument("--index_file", type=str, default="data/embedding/index.faiss", help="The path to save the generated FAISS index")
    parser.add_argument("--document_col_name", type=str, default="content", help="The column name containing the document content")
    parser.add_argument("--device", type=str, default="0", help="The device to use for embedding generation")
    parser.add_argument("--benchmark", type=str, default="bright", help="The benchmark to use")
    parser.add_argument("--dataset", type=str, default="pony", help="The dataset to use")
    parser.add_argument("--version", type=str, default="ori", help="The version of the embeddings, either 'ori' for original documents or 'aug' for augmented documents")
    parser.add_argument("--id_col_name", type=str, default="id", help="The column name containing the document IDs")
    parser.add_argument("--index_type", type=str, default="flat", help="The type of index to use")
    args = parser.parse_args()
    return args




def embed_and_index(args):
    output_dir = f"../../../data_preprocess/eval_data/embeddings/{args.benchmark}/{args.dataset}/{args.version}"
    os.makedirs(output_dir, exist_ok=True)

    model_name = args.model.split("/")[-1]
    
    model = SentenceTransformer(args.model, device=f"cuda:0", trust_remote_code=True)
    
    # Get the actual embedding dimension from the model
    embedding_dim = model.get_sentence_embedding_dimension()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    if args.index_type == "flat":
        index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dim))
    elif args.index_type == "hnsw":
        index = faiss.IndexHNSWFlat(embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64
        index = faiss.IndexIDMap(index)
        
    args.input_file = f"{args.input_file}/{args.dataset}/document.parquet"
    document = pd.read_parquet(args.input_file)
    print(document)

    content = document[args.document_col_name]
    id_list = document[args.id_col_name]

    logging.info(f"dataset: {args.dataset}")
    logging.info(f"using component: {args.document_col_name}")
    logging.info(f"{len(content)} documents found")
    logging.info("=" * 100)

    # build index:id dictionary
    index_id_dict = {i: id_list[i] for i in range(len(id_list))}

    with open(f"{output_dir}/index_id_dict.pkl", "wb") as f:
        pickle.dump(index_id_dict, f)

    logging.info(f"saving index_id_dict")
    logging.info(f"Embedding documents")
    max_seq_length = MODEL_MAX_LEN_DICT.get(args.model, model.get_max_seq_length())

    for i in tqdm(range(0, len(content))):
        c = safe_convert_to_string(content[i])
        
        # Truncate using tokenizer's encode with max_length
        tokens = tokenizer.encode(c, max_length=max_seq_length, truncation=True)
        c = tokenizer.decode(tokens, skip_special_tokens=True)

        result = model.encode(c, show_progress_bar=False)
        embedding = np.array(result, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(embedding)
        index.add_with_ids(embedding, np.array([i], dtype=np.int64))

    faiss.write_index(index, f"{output_dir}/{model_name}_{args.index_type}_index.faiss")
    logging.info(f"Indexing done")



def main():
    args = get_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    embed_and_index(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()