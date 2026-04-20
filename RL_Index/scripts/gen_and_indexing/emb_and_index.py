import faiss
import numpy as np
from vllm import LLM
from argparse import ArgumentParser
from transformers import AutoTokenizer
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
    parser.add_argument("--model", type=str, default="Alibaba-NLP/gte-Qwen1.5-7B-instruct")
    parser.add_argument("--input_file", type=str, default="/anvil/scratch/x-ylei3/NEW_FRAMEWORK/outputs/aug_docs_v2_llama_sbert_multitask")
    parser.add_argument("--index_file", type=str, default="data/embedding/index.faiss")
    parser.add_argument("--document_col_name", type=str, default="aug_content")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--benchmark", type=str, default="bright")
    parser.add_argument("--dataset", type=str, default="pony")
    parser.add_argument("--step", type=int, default=500)
    parser.add_argument("--version", type=str, default="aug")
    parser.add_argument("--id_col_name", type=str, default="id")
    parser.add_argument("--index_type", type=str, default="flat")
    args = parser.parse_args()
    return args




def embed_and_index(args):
    output_dir = f"embeddings/{args.benchmark}/{args.dataset}/{args.version}"
    os.makedirs(output_dir, exist_ok=True)

    model_name = args.model.split("/")[-1]
    
    model = LLM(
        model=args.model,
        device="cuda:0",
        task="embed",
        trust_remote_code=True,
        seed=42
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    if args.index_type == "flat":
        index = faiss.IndexIDMap(faiss.IndexFlatIP(4096))
    elif args.index_type == "hnsw":
        index = faiss.IndexHNSWFlat(4096, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64
        index = faiss.IndexIDMap(index)
        
    args.input_file = f"{args.input_file}/{args.dataset}_{args.step}.parquet"
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

    for i in tqdm(range(0, len(content))):
        c = safe_convert_to_string(content[i])
        tokenized_c = tokenizer.tokenize(c)
        if len(tokenized_c) > MODEL_MAX_LEN_DICT[args.model]:
            tokenized_c = tokenized_c[:MODEL_MAX_LEN_DICT[args.model]]
            c = tokenizer.convert_tokens_to_string(tokenized_c)

        result = model.encode(c, use_tqdm=False)
        embedding = result[0].outputs.embedding
        embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(embedding)
        index.add_with_ids(embedding, np.array([i], dtype=np.int64))

    faiss.write_index(index, f"{output_dir}/LM_{model_name}_{args.index_type}_index.faiss")
    logging.info(f"Indexing done")



def main():
    args = get_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    embed_and_index(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()