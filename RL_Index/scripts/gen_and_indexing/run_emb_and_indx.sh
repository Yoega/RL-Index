step=700
for dataset_name in "biology" "earth_science" "economics" "psychology" "sustainable_living" "robotics" "stackoverflow" "pony" "aops" "theoremqa_questions" "theoremqa_theorems" "leetcode"
do
    CUDA_VISIBLE_DEVICES=0 python emb_and_index.py --dataset $dataset_name --step $step --model "sentence-transformers/all-mpnet-base-v2" \
    --input_file "../../outputs/sbert_llama/aug_docs_v2_llama_sbert" --version "La_Sbert_700"
done

# "Alibaba-NLP/gte-Qwen1.5-7B-instruct", "sentence-transformers/all-mpnet-base-v2", "intfloat/e5-large-v2", BAAI/bge-large-en-v1.5
# "biology" "earth_science" "economics" "psychology" "sustainable_living" "robotics" "stackoverflow" "pony" "aops" "theoremqa_questions" "theoremqa_theorems" "leetcode"