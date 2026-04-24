for dataset in "biology" "earth_science" "economics" "psychology" "sustainable_living" "robotics" "stackoverflow" "pony" "aops" "theoremqa_questions" "theoremqa_theorems" "leetcode"
do
  python eval_combine_LM.py --dataset "$dataset" --step 700 --model "Alibaba-NLP/gte-Qwen1.5-7B-instruct" --version "La_Qwen_RL_700" 
done
# "biology" "earth_science" "economics" "psychology" "sustainable_living" "robotics" "stackoverflow" "pony" "aops" "theoremqa_questions" "theoremqa_theorems" "leetcode"
# for dataset in "biology"
# do
#   python eval_combine_multiemb.py --dataset "$dataset" > biology_analysis.txt
# done
# "Alibaba-NLP/gte-Qwen1.5-7B-instruct"
# BAAI/bge-large-en-v1.5
# sentence-transformers/all-mpnet-base-v2, intfloat/e5-mistral-7b-instruct