python inference_ibm_greedy.py \
    --model_name=meta-llama/llama-3-70b-instruct \
    --batch_size=100 \
    --nprocess=1 \
    --max_new_tokens=128 \
    --prompts_dir=/dccstor/cfm-sempar/long-context-modelling-mLLMs/prompts/prompts_Meta-Llama-3-70B-Instruct_distractor_passages=0 \
    --output_dir=/dccstor/cfm-sempar/long-context-modelling-mLLMs/predictions/Meta-Llama-3-70B-Instruct_distractor_passages=0_greedy \
;

python inference_ibm_greedy.py \
    --model_name=meta-llama/llama-3-70b-instruct \
    --batch_size=100 \
    --nprocess=1 \
    --max_new_tokens=128 \
    --prompts_dir=/dccstor/cfm-sempar/long-context-modelling-mLLMs/prompts/prompts_Meta-Llama-3-70B-Instruct_4k \
    --output_dir=/dccstor/cfm-sempar/long-context-modelling-mLLMs/predictions/Meta-Llama-3-70B-Instruct_4k_greedy \
;

python inference_ibm_greedy.py \
    --model_name=meta-llama/llama-3-70b-instruct \
    --batch_size=100 \
    --nprocess=1 \
    --max_new_tokens=128 \
    --prompts_dir=/dccstor/cfm-sempar/long-context-modelling-mLLMs/prompts/prompts_Meta-Llama-3-70B-Instruct \
    --output_dir=/dccstor/cfm-sempar/long-context-modelling-mLLMs/predictions/Meta-Llama-3-70B-Instruct_greedy \
;

python inference_ibm_greedy.py \
    --model_name=meta-llama/llama-3-70b-instruct \
    --batch_size=100 \
    --nprocess=1 \
    --max_new_tokens=128 \
    --prompts_dir=/dccstor/cfm-sempar/long-context-modelling-mLLMs/prompts/prompts_Meta-Llama-3-70B-Instruct_16k \
    --output_dir=/dccstor/cfm-sempar/long-context-modelling-mLLMs/predictions/Meta-Llama-3-70B-Instruct_16k_greedy \
;

# -------------------------- # # -------------------------- # # -------------------------- # # -------------------------- # # -------------------------- #

python inference_ibm_greedy.py \
    --model_name=mistralai/mixtral-8x7b-instruct-v01 \
    --batch_size=100 \
    --nprocess=1 \
    --max_new_tokens=128 \
    --prompts_dir=/dccstor/cfm-sempar/long-context-modelling-mLLMs/prompts/prompts_Mixtral-8x7B-Instruct-v0.1_distractor_passages=0 \
    --output_dir=/dccstor/cfm-sempar/long-context-modelling-mLLMs/predictions/Mixtral-8x7B-Instruct-v0.1_distractor_passages=0_greedy \
;

python inference_ibm_greedy.py \
    --model_name=mistralai/mixtral-8x7b-instruct-v01 \
    --batch_size=100 \
    --nprocess=1 \
    --max_new_tokens=128 \
    --prompts_dir=/dccstor/cfm-sempar/long-context-modelling-mLLMs/prompts/prompts_Mixtral-8x7B-Instruct-v0.1_4k \
    --output_dir=/dccstor/cfm-sempar/long-context-modelling-mLLMs/predictions/Mixtral-8x7B-Instruct-v0.1_4k_greedy \
;

python inference_ibm_greedy.py \
    --model_name=mistralai/mixtral-8x7b-instruct-v01 \
    --batch_size=100 \
    --nprocess=1 \
    --max_new_tokens=128 \
    --prompts_dir=/dccstor/cfm-sempar/long-context-modelling-mLLMs/prompts/prompts_Mixtral-8x7B-Instruct-v0.1 \
    --output_dir=/dccstor/cfm-sempar/long-context-modelling-mLLMs/predictions/Mixtral-8x7B-Instruct-v0.1_greedy \
;

python inference_ibm_greedy.py \
    --model_name=mistralai/mixtral-8x7b-instruct-v01 \
    --batch_size=100 \
    --nprocess=1 \
    --max_new_tokens=128 \
    --prompts_dir=/dccstor/cfm-sempar/long-context-modelling-mLLMs/prompts/prompts_Mixtral-8x7B-Instruct-v0.1_16k \
    --output_dir=/dccstor/cfm-sempar/long-context-modelling-mLLMs/predictions/Mixtral-8x7B-Instruct-v0.1_16k_greedy \
;
