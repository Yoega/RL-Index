for dataset_name in "leetcode"
do
    CUDA_VISIBLE_DEVICES=0 python emb_and_index.py --dataset $dataset_name --model "sentence-transformers/all-mpnet-base-v2" \
    --input_file "../../../data_preprocess/eval_data/BRIGHT" --version "ori"
done

# "Alibaba-NLP/gte-Qwen1.5-7B-instruct", "sentence-transformers/all-mpnet-base-v2", "intfloat/e5-large-v2"

# "biology" "earth_science" "economics" "psychology" "sustainable_living" "robotics" "stackoverflow" "pony" "aops" "theoremqa_questions" "theoremqa_theorems" "leetcode"