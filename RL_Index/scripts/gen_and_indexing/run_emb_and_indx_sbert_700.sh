step=700
for dataset_name in "biology" "earth_science" "economics" "psychology" "sustainable_living" "robotics" "stackoverflow" "pony" "aops" "theoremqa_questions" "theoremqa_theorems" "leetcode"
do
    CUDA_VISIBLE_DEVICES=0 python emb_and_index.py --dataset $dataset_name --step $step --model "Alibaba-NLP/gte-Qwen1.5-7B-instruct" \
    --input_file "/anvil/scratch/x-ylei3/NEW_FRAMEWORK/outputs/aug_docs_v2_qwen_qwen" --version "Qw_Qwen_RL_$step"
done

# "biology" "earth_science" "economics" "psychology" "sustainable_living" "robotics" "stackoverflow" "pony" "aops" "theoremqa_questions" "theoremqa_theorems" "leetcode"
## "Alibaba-NLP/gte-Qwen1.5-7B-instruct", "sentence-transformers/all-mpnet-base-v2"