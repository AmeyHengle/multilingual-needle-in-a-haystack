"""
nohup python generate_prompts.py \
    --experiments_file /home/amey/long-context-llms/experiments/experiment0_baseline.json \
    --output_dir /home/amey/long-context-llms/prompts/prompts_distractor_passages=0 \
    --model_name /home/models/Llama-2-7b-chat-hf \
    --max_seq_len 4096 \
    --test_size 100 \
    ;
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from prompt_templates import (ICE_template1, ICE_template2, prompt_template1,
                              prompt_template2, prompt_template3,
                              prompt_template4)
from utils import (dashed_line, experiment_dict_to_fname,
                   fname_to_experiment_dict, get_top_n_keys, load_json,
                   merge_dicts, save_json)

from preprocess import Preprocess

data_mlqa = pd.read_csv("data/mlqa_combined_formatted.csv")
print(dashed_line)
print(f"Total dataset (MLQA): {data_mlqa.shape[0]}")
print(dashed_line)

corpus = load_json("data/ss_mlqa_msmarco_mpnet.json")
print(dashed_line)
print(f"Total corpus (semantic similarity): {len(corpus)}")
print(dashed_line)


def insert_needle(needle, distractor_passages, position):
    """
    Inserts the input_string into the string_list based on the specified position.

    Args:
        input_string (str): The string to be inserted.
        string_list (list): The list of strings.
        position (str): The position where input_string should be inserted ('start', 'end', or 'middle').

    Returns:
        list: The updated string_list.
    """
    if position == "start":
        # Insert at the beginning (index 0)
        distractor_passages.insert(0, needle)
    elif position == "end":
        # Insert at the end
        distractor_passages.append(needle)
    elif position == "middle":
        # Insert in the middle (after the first element)
        middle_index = len(distractor_passages) // 2
        distractor_passages.insert(middle_index, needle)
    else:
        print("Invalid position. Please choose 'start', 'end', or 'middle'.")

    return distractor_passages


def get_distractor_passages_ss(
    noise_type, matches, num_docs=20, similarity_threshold=0.1
) -> list[str]:
    if noise_type == "multilingual":
        merged_dict = merge_dicts([v for k, v in matches.items()])
        top_matches = get_top_n_keys(
            input_dict=merged_dict, n=num_docs, threshold=similarity_threshold
        )
    else:
        assert noise_type in matches
        top_matches = get_top_n_keys(
            input_dict=matches[noise_type], n=num_docs, threshold=similarity_threshold
        )
    return top_matches


def generate_prompts(
    # data: pd.DataFrame,
    experiment_dict: dict,
    model_name: str,
    max_seq_len: int,
    test_size: int,
    prompt_template,
    ice_template,
) -> list[str]:
    """
    Given an experiment, this code returns a list of prompts for LLM inference
    """
    preprocessing_pipeline = Preprocess(model_name=model_name, max_seq_len=max_seq_len)
    context_lang = experiment_dict["context_lang"]
    question_lang = experiment_dict["question_lang"]
    noise_type = experiment_dict["noise_type"]
    retrieval_type = experiment_dict["retrieval_type"]
    needle_position = experiment_dict["needle_position"]
    prompts = []

    df = data_mlqa.copy()
    df = df[
        (df["context_lang"] == context_lang) & (df["question_lang"] == question_lang)
    ]  # Filter relevant instances
    df = df[:test_size]

    print(f"Total queries: {df.shape[0]}")
    print(dashed_line)
    assert df.shape[1] > 0
    for i, row in tqdm(df.iterrows(), desc="Generating prompts"):
        # Noise retrieval code - retrieve passages from data to be appended to the prompt along with the needle.
        needle_passage = row["context"]
        context_id = row["id"]

        distractor_passages = retrieve_distractor_passages(
            # data=data,
            context_id=context_id,
            # needle_passage=needle_passage,
            context_lang=context_lang,
            retrieval_type=retrieval_type,
            noise_type=noise_type,
        )
        distractor_passages = preprocessing_pipeline.preprocess(
            needle_passage=needle_passage,
            distractor_passages=distractor_passages,
            prompt_template=prompt_template,
        )
        context_passages = insert_needle(
            needle=needle_passage,
            distractor_passages=distractor_passages,
            position=needle_position,
        )
        context = ice_template(context_passages)
        context = context.strip()

        question = row["question"]
        prompt = prompt_template(question, context)
        prompts.append(prompt)

    # return prompts
    df["prompt"] = prompts
    return df


def prasoon_to_amey(matches):
    lang_dict = {
        "english": "en",
        "hindi": "hi",
        "arabic": "ar",
        "vietnamese": "vi",
        "chinese": "zh",
        "spanish": "es",
        "german": "de",
    }

    matches_dict = {}
    for key, value in matches.items():
        matches_dict[lang_dict[key]] = {}
        for x in value:
            matches_dict[lang_dict[key]][x[0]] = x[1]
    return matches_dict


def retrieve_distractor_passages(
    # data_mlqa: pd.DataFrame,
    context_id,
    context_lang,
    retrieval_type,
    noise_type,
    num_paras=0,
):
    assert retrieval_type in ["random", "semantic_similarity"]
    distractor_passages = []

    if retrieval_type == "random":

        if noise_type == "multilingual":
            distractor_passages = (
                data_mlqa[data_mlqa["context_lang"] != context_lang]
                .sample(num_paras)["context"]
                .values.tolist()
            )

        else:
            distractor_passages = (
                data_mlqa[data_mlqa["context_lang"] == noise_type]
                .sample(num_paras)["context"]
                .values.tolist()
            )

    elif retrieval_type == "semantic_similarity":
        try:
            matches = corpus[context_lang][context_id]
            matches_dict = prasoon_to_amey(matches)
            distractor_passages = get_distractor_passages_ss(
                noise_type=noise_type,
                matches=matches_dict,
            )
        except KeyError:
            print(f"Match not found")
            print(f"ID: {context_id}")
            print(f"Context Lang: {context_lang}")

    return distractor_passages


def main(
    experiments_file,
    output_dir,
    model_name,
    max_seq_len,
    test_size,
    save_cols=[
        "id",
        "context_lang",
        "question_lang",
        "answer_text_format",
        "answer_start_index",
        "answer_sentence",
        "prompt",
    ],
):
    experiments = load_json(experiments_file)
    print(f"Total experiments: {len(experiments)}")
    print(dashed_line)

    for i, experiment in tqdm(enumerate(experiments), desc="Generating prompts."):
        print(f"Generating prompt for experiment {i}: {experiment}")

        df_prompt = generate_prompts(
            # data=data,
            experiment_dict=experiment,
            model_name=model_name,
            max_seq_len=max_seq_len,
            test_size=test_size,
            prompt_template=prompt_template4,
            ice_template=ICE_template2,
        )

        outfile_name = experiment_dict_to_fname(experiment, extension="csv")
        df_prompt[save_cols].to_csv(os.path.join(output_dir, outfile_name), index=False)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Run the main function with command line arguments."
    )

    # Add the arguments
    parser.add_argument(
        "--experiments_file",
        type=str,
        required=True,
        help="The path to the experiments file.",
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
        "--max_seq_len", type=int, required=False, help="The maximum sequence length."
    )
    parser.add_argument("--test_size", type=int, required=True, help="The test size.")

    # Parse the arguments
    args = parser.parse_args()
    if args.max_seq_len == None or args.max_seq_len == "":
        args.max_seq_len = None

    # Run the main function with the command line arguments
    main(
        args.experiments_file,
        args.output_dir,
        args.model_name,
        args.max_seq_len,
        args.test_size,
    )
