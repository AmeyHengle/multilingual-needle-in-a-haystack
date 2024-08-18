import argparse
import gc
import glob
import os

import pandas as pd
import torch
import transformers
from tqdm import tqdm

from utils import (dashed_line, experiment_dict_to_fname,
                   fname_to_experiment_dict, load_json, save_json)


def free_memory():
    torch.cuda.empty_cache()
    gc.collect()


def postprocess_aya_output(output):
    return [response[0]["generated_text"][1]["content"] for response in output]


def format_inputs(prompts):
    formatted_inputs = []
    for prompt in prompts:
        formatted_inputs.append([{"role": "user", "content": f"{prompt}"}])
    return formatted_inputs


def parse_predictions(predictions):
    predictions_list = []
    for prediction in predictions:
        try:
            predictions_list.append(prediction[0]["generated_text"][1]["content"])
        except Exception as e:
            print(f"Exception parsing model output: {prediction}")
            print(e)
            print(dashed_line)
            predictions_list.append(prediction)

    return predictions_list


def inference(prompts, pipeline, batch_size=32):
    free_memory()
    predictions = []

    for i in tqdm(
        range(0, len(prompts), batch_size),
        desc=f"Running inference on {len(prompts)} data points",
    ):
        batch_inputs = prompts[i : i + batch_size]
        batch_inputs = format_inputs(batch_inputs)
        preds = pipeline(batch_inputs)
        predictions_parsed = parse_predictions(preds)
        predictions.extend(predictions_parsed)

    return predictions


if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(
        description="Run the main function with command line arguments."
    )

    # Add the arguments
    parser.add_argument(
        "--prompts_file",
        type=str,
        required=False,
        help="The path to the prompts file.",
    )
    parser.add_argument(
        "--prompts_dir",
        type=str,
        required=False,
        help="The directory to the experiments file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The path to the output directory.",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="The name of the model."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="The cumulative probability for nucleus sampling (top-p). Default is 1.0.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="The number of highest probability vocabulary tokens to keep for top-k sampling. Default is 50.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="The value to control the randomness of predictions by scaling the logits before applying softmax. Default is 0.7.",
    )
    parser.add_argument("--batch_size", type=int, required=True, help="The batch size.")
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "float64"],
        help="The torch dtype to use during inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="The device to use for inference ('cuda' or 'cpu').",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Setup the tokenizer and pipeline
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model_name,
        tokenizer=tokenizer,
        torch_dtype=getattr(torch, args.torch_dtype),
        trust_remote_code=True,
        device=device,
        do_sample=True,
        temperature=0.1,
        top_p=1,
        top_k=50,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=128,
    )

    if args.prompts_dir is not None:
        filenames = glob.glob(f"{args.prompts_dir}/*.csv")
        filenames = [
            filename for filename in filenames if "needle_position:end" in filename
        ]
        print(f"Total experiments: {len(filenames)}")
        for filename in filenames:
            prompts_df = pd.read_csv(filename)
            if f"predictions_{args.model_name}" not in prompts_df.columns:
                print(f"Running inference for {filename}")
                print(dashed_line)
                prompts = prompts_df["prompt"].values.tolist()

                # Run the inference function with the command line arguments
                predictions = inference(
                    prompts=prompts, pipeline=pipeline, batch_size=args.batch_size
                )

                # Save predictions to specified output dir
                assert len(predictions) == prompts_df.shape[0]
                output_path_csv = os.path.join(
                    args.output_dir, os.path.basename(filename)
                )
                output_path_pkl = output_path_csv.replace(".csv", ".pkl")
                prompts_df[f"predictions_{args.model_name}"] = predictions
                prompts_df.to_csv(output_path_csv, index=False)
                prompts_df.to_pickle(output_path_pkl)
            else:
                print(f"Skipping file {filename}, predictions already present")
