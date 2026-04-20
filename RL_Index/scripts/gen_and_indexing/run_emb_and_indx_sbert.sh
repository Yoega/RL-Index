step=700
for dataset_name in "biology" "earth_science" "economics" "psychology" "sustainable_living" "robotics" "stackoverflow" "pony" "aops" "theoremqa_questions" "theoremqa_theorems" "leetcode"
do
    CUDA_VISIBLE_DEVICES=0 python emb_and_index_LM.py --dataset $dataset_name --step $step --model "Alibaba-NLP/gte-Qwen1.5-7B-instruct" \
    --input_file "../../outputs/aug_docs" --version "Qw_Qwen_RL_$step"
done
