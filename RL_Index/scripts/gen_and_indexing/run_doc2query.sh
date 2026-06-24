# # define a variable step step = 1000
# for dataset_name in "leetcode"
# do
#     CUDA_VISIBLE_DEVICES=0 python doc2query.py --model_path macavaney/doc2query-t5-base-msmarco \
#         --input_path ../../data_preprocess/eval_data/BRIGHT \
#         --output_path ../../outputs/aug_docs/${dataset_name}_doc2query_10.parquet \
#         --dataset_name $dataset_name \
#         --rewritten_content document \
#         --batch_size 16 \
#         --num_queries 10
# done
# # # # "biology" "earth_science" "economics" "psychology" "sustainable_living" "robotics" "stackoverflow" "pony" "aops" "theoremqa_questions" "theoremqa_theorems" "leetcode"


step="doc2query_10"
for dataset_name in "leetcode"
do
    CUDA_VISIBLE_DEVICES=0 python emb_and_index.py --dataset $dataset_name --step $step --model "BAAI/bge-large-en-v1.5" \
    --input_file "../../outputs/aug_docs" --version "Doc2Query_10"
done

# "Alibaba-NLP/gte-Qwen1.5-7B-instruct", "sentence-transformers/all-mpnet-base-v2", "intfloat/e5-large-v2", BAAI/bge-large-en-v1.5
# "biology" "earth_science" "economics" "psychology" "sustainable_living" "robotics" "stackoverflow" "pony" "aops" "theoremqa_questions" "theoremqa_theorems" "leetcode"