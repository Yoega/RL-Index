import logging
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


scoring_model_path = "sentence-transformers/all-mpnet-base-v2" # BAAI/bge-large-en-v1.5,sentence-transformers/all-mpnet-base-v2, "Alibaba-NLP/gte-Qwen1.5-7B-instruct", intfloat/e5-mistral-7b-instruct
scoring_model = SentenceTransformer(scoring_model_path, device='cpu')


def extract_aug_text(solution_str: str, use_think_flag: bool) -> str:
    """
    Extract the actual query text from a solution string.
    
    - If `<|im_start|>assistant` exists, strip everything before it.
    - If `think_flag` is True, extract the text inside <answer>...</answer> 
      after optional <think>...</think>.
    """
    aug = solution_str

    # Remove system prompt part if exists
    if "<|im_start|>assistant" in aug:
        aug = aug.split("<|im_start|>assistant", 1)[1]

    # Extract <answer>...</answer> if think_flag is set
    if use_think_flag:
        pattern = re.compile(r"<think>.*?</think>\s*<answer>(.*?)</answer>", re.DOTALL)
        match = pattern.search(aug)
        if not match:
            return ""  # Return empty string if no match found
        aug = match.group(1)

    return aug.strip()


def compute_score(aug_str: str, ground_truth: str) -> float:
    """
    Compute the reward score.

    Args:
        solution_str (str): The generated solution string.
        ground_truth (str): JSON string with fields:
            - "query": original query
            - "pos": positive document
            - "neg": negative document
            - "think_flag": whether <think>/<answer> formatting should be used

    Returns:
        float: Advantage score (scaled difference between query and original question similarity).
               Returns -1 if parsing fails.
    """
    try:
        # Parse the ground truth JSON
        label_item = json.loads(ground_truth)
        query = label_item["query"]
        pos_doc = label_item["pos"]
        think_flag = label_item.get("think_flag", False)

        if think_flag:
            aug_str = extract_aug_text(aug_str, think_flag)
            
        if not aug_str:
            return -1
        
        device = 'cuda'
        scoring_model.to(device)
        # Compute embeddings
        query_emb = scoring_model.encode([query], normalize_embeddings=True)
        ori_pos_doc_emb = scoring_model.encode([pos_doc], normalize_embeddings=True)
        aug_pos_doc_emb = scoring_model.encode([aug_str], normalize_embeddings=True)
        
        # scoring_model.to('cpu')
        # torch.cuda.empty_cache()

        # Cosine similarity (dot product since embeddings are normalized)
        ori_pos_sim = query_emb @ ori_pos_doc_emb.T
        aug_pos_sim = query_emb @ aug_pos_doc_emb.T

        # Compute advantage score
        advantage = float((aug_pos_sim - ori_pos_sim) * 10)
        logging.info("Got Advantage:%s", advantage)
        return advantage

    except Exception as e:
        logging.info(f"[Error in compute_score] {e}")
        return -1


def reward_func(completions, answer, **kwargs):
    """
    Reward function that computes embedding-model-based scores for multiple completions.

    Args:
        completions (list): List of completions, each a list of dicts with 'content'.
        answers (list): List of ground truth JSON strings.
        kwargs: Unused, for compatibility.

    Returns:
        list: List of reward scores for each (completion, answer) pair.
    """
    responses = [completion[0]["content"] for completion in completions]
    return [compute_score(r, a) for r, a in zip(responses, answer)]