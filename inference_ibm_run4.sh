python inference_ibm.py \
    --model_name=meta-llama/llama-3-70b-instruct \
    --batch_size=100 \
    --nprocess=1 \
    --max_new_tokens=128 \
    --prompts_dir=prompts/prompts_missing_2/prompts_Meta-Llama-3-70B-Instruct_4k \
    --output_dir=predictions/Meta-Llama-3-70B-Instruct_4k \
;

python inference_ibm.py \
    --model_name=meta-llama/llama-3-70b-instruct \
    --batch_size=100 \
    --nprocess=1 \
    --max_new_tokens=128 \
    --prompts_dir=prompts/prompts_missing_2/prompts_Meta-Llama-3-70B-Instruct \
    --output_dir=predictions/Meta-Llama-3-70B-Instruct \
;

# -------------------------- #

python inference_ibm.py \
    --model_name=mistralai/mistral-7b-instruct-v0-2 \
    --batch_size=100 \
    --nprocess=1 \
    --max_new_tokens=128 \
    --prompts_dir=prompts/prompts_missing_2/prompts_Mistral-7B-Instruct-v0.2 \
    --output_dir=predictions/Mistral-7B-Instruct-v0.2 \
;

python inference_ibm.py \
    --model_name=mistralai/mistral-7b-instruct-v0-2 \
    --batch_size=100 \
    --nprocess=1 \
    --max_new_tokens=128 \
    --prompts_dir=prompts/prompts_missing_2/prompts_Mistral-7B-Instruct-v0.2_4k \
    --output_dir=predictions/Mistral-7B-Instruct-v0.2_4k \
;

python inference_ibm.py \
    --model_name=mistralai/mistral-7b-instruct-v0-2 \
    --batch_size=100 \
    --nprocess=1 \
    --max_new_tokens=128 \
    --prompts_dir=prompts/prompts_missing_2/prompts_Mistral-7B-Instruct-v0.2_8k \
    --output_dir=predictions/Mistral-7B-Instruct-v0.2_8k \
;

python inference_ibm.py \
    --model_name=mistralai/mistral-7b-instruct-v0-2 \
    --batch_size=100 \
    --nprocess=1 \
    --max_new_tokens=128 \
    --prompts_dir=prompts/prompts_missing_2/prompts_Mistral-7B-Instruct-v0.2_16k \
    --output_dir=predictions/Mistral-7B-Instruct-v0.2_16k \
;


# -------------------------- #

python inference_ibm.py \
    --model_name=mistralai/mixtral-8x7b-instruct-v01 \
    --batch_size=100 \
    --nprocess=1 \
    --max_new_tokens=128 \
    --prompts_dir=prompts/prompts_missing_2/prompts_Mixtral-8x7B-Instruct-v0.1_4k \
    --output_dir=predictions/Mixtral-8x7B-Instruct-v0.1_4k \
;

python inference_ibm.py \
    --model_name=mistralai/mixtral-8x7b-instruct-v01 \
    --batch_size=100 \
    --nprocess=1 \
    --max_new_tokens=128 \
    --prompts_dir=prompts/prompts_missing_2/prompts_Mixtral-8x7B-Instruct-v0.1 \
    --output_dir=predictions/Mixtral-8x7B-Instruct-v0.1 \
;