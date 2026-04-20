step=600
for dataset_name in "biology" "earth_science" "economics" "psychology" "sustainable_living" "robotics" "stackoverflow" "pony" "aops" "theoremqa_questions" "theoremqa_theorems" "leetcode"
do
    CUDA_VISIBLE_DEVICES=0 python emb_and_index.py --dataset $dataset_name --step $step --model "BAAI/bge-large-en-v1.5" \
    --input_file "/anvil/scratch/x-ylei3/NEW_FRAMEWORK/outputs/aug_docs_v2_llama_sbert" --version "RL_$step"
done

# "biology" "earth_science" "economics" "psychology" "sustainable_living" "robotics" "stackoverflow" "pony" "aops" "theoremqa_questions" "theoremqa_theorems" "leetcode"

#