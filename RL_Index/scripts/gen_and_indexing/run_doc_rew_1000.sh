# define a variable step step = 800
step=1000
for dataset_name in "biology" "earth_science" "economics" "psychology" "sustainable_living" "robotics" "stackoverflow" "pony" "aops" "theoremqa_questions" "theoremqa_theorems" "leetcode"
do
    CUDA_VISIBLE_DEVICES=0 python document_rewriting.py --model_path /anvil/scratch/x-ylei3/NEW_FRAMEWORK/outputs/RL_qwen_Qwen2.5-1.5B-Instruct_v2_train_no_think_seed5202_maxcomp500_numgen16/checkpoint-$step \
        --input_path /anvil/scratch/x-ylei3/eval_sft/data/BRIGHT \
        --output_path /anvil/scratch/x-ylei3/NEW_FRAMEWORK/outputs/aug_docs_v2_qwen_qwen/${dataset_name}_${step}.parquet \
        --dataset_name $dataset_name \
        --rewritten_content document
done

# "biology" "earth_science" "economics" "psychology" "sustainable_living" "robotics" "stackoverflow" "pony" "aops" "theoremqa_questions" "theoremqa_theorems" "leetcode"

# python document_rewriting.py --model_path /anvil/scratch/x-ylei3/NEW_FRAMEWORK/outputs/all_no_think_seed5202_maxcomp800_numgen16/checkpoint-200 \
#         --input_path /anvil/scratch/x-ylei3/train_evaluation/data/bio_train_doc_df.parquet \
#         --output_path /anvil/scratch/x-ylei3/train_evaluation/data/aug_docs/bio_train.parquet \
#         --dataset_name bio_train


# python document_rewriting.py --model_path /anvil/scratch/x-ylei3/NEW_FRAMEWORK/outputs/all_no_think_seed5202_maxcomp500_numgen16/checkpoint-500 \
#         --input_path /anvil/scratch/x-ylei3/eval_sft/data/BRIGHT \
#         --output_path /anvil/scratch/x-ylei3/NEW_FRAMEWORK/outputs/aug_docs/aops.parquet \
#         --dataset_name aops