python inference_ibm.py \
    --model_name=meta-llama/llama-3-70b-instruct \
    --batch_size=100 \
    --nprocess=1 \
    --max_new_tokens=128 \
    --prompts_dir=prompts/Meta-Llama-3-70B-Instruct \
    --output_dir=predictions/Meta-Llama-3-70B-Instruct \
;


python inference_ibm.py \
    --model_name=mistralai/mistral-7b-instruct-v0-2 \
    --batch_size=100 \
    --nprocess=1 \
    --max_new_tokens=128 \
    --prompts_dir=prompts/Mistral-7B-Instruct-v0.2 \
    --output_dir=predictions/Mistral-7B-Instruct-v0.2 \
;

python inference_ibm.py \
    --model_name=mistralai/mixtral-8x7B-instruct \
    --batch_size=100 \
    --nprocess=1 \
    --max_new_tokens=128 \
    --prompts_dir=prompts/Mixtral-8x7B-Instruct-v0.1 \
    --output_dir=predictions/Mixtral-8x7B-Instruct-v0.1 \
;

