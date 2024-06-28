import argparse
import gc
import glob
import os
import sys
import time
import pandas as pd
import torch
import transformers
from tqdm import tqdm

from utils import dashed_line, load_json, save_json


def free_memory():
    torch.cuda.empty_cache()
    gc.collect()


# import _settings
import dataeval.coqa as coqa
import dataeval.nq_open as nq_open
import dataeval.squad_reader as squad
import dataeval.triviaqa as triviaqa
from dotenv import load_dotenv
from genai.client import Client
from genai.credentials import Credentials
from genai.schema import (DecodingMethod, TextGenerationParameters,
                          TextGenerationReturnOptions)

# Create the parser
parser = argparse.ArgumentParser(
    description="Run the main function with command line arguments."
)

# Add the arguments
parser.add_argument(
    "--prompts_dir",
    type=str,
    required=True,
    help="The directory to the prompts file.",
)

parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="The directory to the store predictions file.",
)

parser.add_argument(
    "--model_name", type=str, required=True, help="The name of the model."
)

parser.add_argument("--batch_size", type=int, required=True, help="The batch size.")
parser.add_argument(
    "--max_new_tokens", type=int, default=128, help="Maximum new tokens to generate"
)
parser.add_argument("--num_generations_per_prompt", type=int, default=1)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--decoding_method", type=str, default="sample")
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--nprocess", type=int, default=None)

# Parse the arguments
args = parser.parse_args()

print(args)


def parse_predictions(repsonses):
    predictions_list = []
    for prediction in repsonses:
        try:
            predictions_list.append(prediction[0]["generated_text"])
        except Exception as e:
            print(f"Exception parsing model output: {prediction}")
            print(e)
            print(dashed_line)
            predictions_list.append(prediction)

    return predictions_list


def get_generations_bam(
    model_name: str,
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int,
    args,
    seed=42,
):
    load_dotenv(dotenv_path=".env")
    client = Client(credentials=Credentials.from_env())
    predictions = []

    parameters = TextGenerationParameters(
        decoding_method=DecodingMethod.SAMPLE,
        max_new_tokens=max_new_tokens,
        return_options=TextGenerationReturnOptions(generated_tokens=True),
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=1.5,
        stop_sequences=["()"],
        random_seed=args.seed,
    )

    for i in tqdm(
        range(0, len(prompts), batch_size),
        desc=f"Running inference on {len(prompts)} data points",
    ):
        batch_inputs = prompts[i : i + batch_size]
        responses = client.text.generation.create(
            model_id=model_name, inputs=batch_inputs, parameters=parameters
        )
        batch_preds = [response.results[0].generated_text for response in responses]
        predictions.extend(batch_preds)

    return predictions


def main():
    if args.prompts_dir != None:
        for filename in glob.glob(f"{args.prompts_dir}/*.csv"):
            outfile = os.path.join(args.output_dir, filename.split("/")[-1])
            if os.path.exists(outfile): # Skip existing predictions
                print(f"Skipping {filename}, predictions exist.")

            prompts_df = pd.read_csv(filename)
            if (
                f"predictions_{args.model_name}" not in prompts_df.columns
            ):  # Skip files with predictions columns already populaed
                print(f"Running inference for {filename}")
                print(dashed_line)
                prompts = prompts_df["prompt"].values.tolist()

                # Run the main function with the command line arguments (Num of retires = 5)
                num_retries = 5
                retry_delay = 900  # 15 minutes in seconds

                for i in range(num_retries):
                    try:
                        predictions = get_generations_bam(
                            model_name=args.model_name,
                            prompts=prompts,
                            max_new_tokens=args.max_new_tokens,
                            batch_size=args.batch_size,
                            args=args,
                        )
                        break  # Exit the loop if the operation is successful
                    except Exception as e:
                        if i == num_retries - 1:
                            print(f"Maximum number of retries ({num_retries}) reached. Aborting.")
                            raise e  # Re-raise the exception after the last retry
                        else:
                            print(f"Operation failed on attempt {i+1}. Retrying in {retry_delay/60:.2f} minutes...")
                            time.sleep(retry_delay)
                            continue

                # Save predictions to specified output dir
                assert len(predictions) == prompts_df.shape[0]
                prompts_df[f"predictions_{args.model_name}"] = predictions
                prompts_df[["id", f"predictions_{args.model_name}"]].to_csv(
                    os.path.join(args.output_dir, filename.split("/")[-1]), index=False
                )
    return


if __name__ == "__main__":
    # task_runner = main(parallel=args.nprocess)
    main()

### Sample Usage
"""
python inference_ibm.py \
    --model_name=mistralai/mixtral-8x7B-instruct \
    --batch_size=100 \
    --nprocess=1 \
    --max_new_tokens=128 \
    --prompts_dir=prompts/Mixtral-8x7B-Instruct-v0.1 \
    --output_dir=predictions/Mixtral-8x7B-Instruct-v0.1 \
;
"""
