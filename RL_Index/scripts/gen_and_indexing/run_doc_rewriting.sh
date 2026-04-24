# define a variable step step = 1000
step=1000
for dataset_name in "biology" "earth_science" "economics" "psychology" "sustainable_living" "robotics" "stackoverflow" "pony" "aops" "theoremqa_questions" "theoremqa_theorems" "leetcode"
do
    CUDA_VISIBLE_DEVICES=0 python document_rewriting.py --model_path ../../outputs/sbert_qwen/checkpoint-$step \
        --input_path ../../data_preprocess/eval_data/BRIGHT \
        --output_path ../../outputs/aug_docs/${dataset_name}_${step}.parquet \
        --dataset_name $dataset_name \
        --rewritten_content document
done

# "biology" "earth_science" "economics" "psychology" "sustainable_living" "robotics" "stackoverflow" "pony" "aops" "theoremqa_questions" "theoremqa_theorems" "leetcode"