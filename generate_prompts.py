import argparse
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocess import Preprocess
from prompt_templates import (prompt_template_exact_accuracy,
                              prompt_template_existence_accuracy,
                              prompt_template_ICE)
from utils import (dashed_line, experiment_dict_to_fname,
                   fname_to_experiment_dict, get_top_n_keys, load_json,
                   merge_dicts, save_json)

assert (
    "HF_TOKEN" in os.environ
), "Hugging Face token ('HF_TOKEN') not set in environment variables."

# Load datasets
data_mlqa = pd.read_csv("data/mlqa_combined_formatted.csv")
print(dashed_line)
print(f"Total dataset (MLQA): {data_mlqa.shape[0]}")
print(dashed_line)

# Load the semantic similarity corpus
corpus = load_json("data/new_ss_mlqa_msmarco_mpnet_top300.json")
print(dashed_line)
print(f"Total corpus (semantic similarity): {len(corpus)}")
print(dashed_line)


def insert_needle(
    needle: str, distractor_passages: list[str], position: str
) -> list[str]:
    """
    Inserts the needle (important passage) into the list of distractor passages based on the specified position.

    Args:
        needle (str): The passage to be inserted.
        distractor_passages (list): The list of distractor passages.
        position (str): The position where the needle should be inserted ('start', 'end', or 'middle').

    Returns:
        list[str]: The updated list of passages including the needle.
    """
    if position == "start":
        distractor_passages.insert(0, needle)
    elif position == "end":
        distractor_passages.append(needle)
    elif position == "middle":
        middle_index = len(distractor_passages) // 2
        distractor_passages.insert(middle_index, needle)
    else:
        print("Invalid position. Please choose 'start', 'end', or 'middle'.")

    return distractor_passages


def get_distractor_passages_ss(
    noise_type: str,
    matches: dict,
    num_docs: int = 500,
    similarity_threshold: float = 0.01,
) -> list[str]:
    """
    Retrieves the top distractor passages based on semantic similarity.

    Args:
        noise_type (str): The type of noise to consider (e.g., 'multilingual').
        matches (dict): A dictionary containing the matching passages.
        num_docs (int): The number of top distractor passages to retrieve.
        similarity_threshold (float): The minimum similarity threshold for considering passages.

    Returns:
        list[str]: A list of top distractor passages.
    """
    if noise_type == "multilingual":
        merged_dict = merge_dicts([v for k, v in matches.items()])
        top_matches = get_top_n_keys(
            input_dict=merged_dict, n=num_docs, threshold=similarity_threshold
        )
    else:
        assert noise_type in matches, f"Noise type '{noise_type}' not found in matches."
        top_matches = get_top_n_keys(
            input_dict=matches[noise_type], n=num_docs, threshold=similarity_threshold
        )

    return top_matches


def generate_prompts(
    experiment_dict: dict,
    model_name: str,
    max_seq_len: int,
    test_size: int,
    prompt_template,
    ice_template,
) -> pd.DataFrame:
    """
    Generates a list of prompts for large language model (LLM) inference based on the experiment setup.

    Args:
        experiment_dict (dict): Dictionary containing the details of the experiment.
        model_name (str): Name of the model to use.
        max_seq_len (int): Maximum sequence length for the model.
        test_size (int): Number of samples to generate prompts for.
        prompt_template (function): Function to generate the prompt template.
        ice_template (function): Function to generate the in-context example template.

    Returns:
        pd.DataFrame: A DataFrame containing the generated prompts along with other relevant details.
    """
    preprocessing_pipeline = Preprocess(model_name=model_name, max_seq_len=max_seq_len)
    context_lang = experiment_dict["context_lang"]
    question_lang = experiment_dict["question_lang"]
    noise_type = experiment_dict["noise_type"]
    retrieval_type = experiment_dict["retrieval_type"]
    needle_position = experiment_dict["needle_position"]

    df = data_mlqa[
        (data_mlqa["context_lang"] == context_lang)
        & (data_mlqa["question_lang"] == question_lang)
    ].copy()[
        :test_size
    ]  # Filter relevant instances and limit to test size

    print(f"Total queries: {df.shape[0]}")
    print(dashed_line)
    assert df.shape[1] > 0, "DataFrame has no columns."

    prompts = []

    for i, row in tqdm(df.iterrows(), desc="Generating prompts"):
        needle_passage = row["context"]
        context_id = row["id"]

        # Retrieve distractor passages based on noise and retrieval type
        distractor_passages = retrieve_distractor_passages(
            context_id=context_id,
            context_lang=context_lang,
            retrieval_type=retrieval_type,
            noise_type=noise_type,
        )

        # Preprocess and truncate distractor passages if needed
        distractor_passages = preprocessing_pipeline.preprocess(
            needle_passage=needle_passage,
            distractor_passages=distractor_passages,
            prompt_template=prompt_template,
        )

        # Insert the needle passage into the context
        context_passages = insert_needle(
            needle=needle_passage,
            distractor_passages=distractor_passages,
            position=needle_position,
        )
        context = ice_template(context_passages).strip()

        question = row["question"]
        prompt = prompt_template(question, context)
        prompts.append(prompt)

    df["prompt"] = prompts
    return df


def language_mapping(matches: dict) -> dict:
    """
    Maps languages from their full names to their ISO codes in the matches dictionary.

    Args:
        matches (dict): A dictionary with language names as keys and their respective data as values.

    Returns:
        dict: A dictionary with ISO language codes as keys and their respective data as values.
    """
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
        iso_code = lang_dict.get(key)
        if iso_code:
            matches_dict[iso_code] = {x[0]: x[1] for x in value}

    return matches_dict


def retrieve_distractor_passages(
    context_id: str,
    context_lang: str,
    retrieval_type: str,
    noise_type: str,
    num_paras: int = 0,
) -> list[str]:
    """
    Retrieves distractor passages based on the retrieval type and noise type.

    Args:
        context_id (str): The ID of the context passage.
        context_lang (str): The language of the context passage.
        retrieval_type (str): The type of retrieval to perform ('random' or 'semantic_similarity').
        noise_type (str): The type of noise to consider (e.g., 'multilingual').
        num_paras (int): The number of paragraphs to retrieve.

    Returns:
        list[str]: A list of retrieved distractor passages.
    """
    assert retrieval_type in [
        "random",
        "semantic_similarity",
    ], "Invalid retrieval type."

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
            matches_dict = language_mapping(matches)
            distractor_passages = get_distractor_passages_ss(
                noise_type=noise_type, matches=matches_dict, num_docs=500
            )
        except KeyError:
            print(f"Match not found for ID: {context_id} in language: {context_lang}")

    return distractor_passages


def main(
    experiments_file: str,
    output_dir: str,
    model_name: str,
    max_seq_len: int,
    test_size: int,
    evaluation_metric: str,
    save_cols: list[str] = [
        "id",
        "context_lang",
        "question_lang",
        "answer_text_format",
        "answer_start_index",
        "answer_sentence",
        "prompt",
    ],
):
    """
    Main function to generate prompts based on the experiments provided.

    Args:
        experiments_file (str): Path to the JSON file containing experiment configurations.
        output_dir (str): Directory to save the output CSV files.
        model_name (str): The name of the model to use.
        max_seq_len (int): The maximum sequence length for the model.
        test_size (int): The number of test samples to process.
        save_cols (list[str]): The columns to save in the output CSV file.
    """
    experiments = load_json(experiments_file)
    print(f"Total experiments: {len(experiments)}")
    print(dashed_line)

    for i, experiment in tqdm(enumerate(experiments), desc="Generating prompts"):
        print(f"Generating prompt for experiment {i}: {experiment}")

        df_prompt = generate_prompts(
            experiment_dict=experiment,
            model_name=model_name,
            max_seq_len=max_seq_len,
            test_size=test_size,
            prompt_template=(
                prompt_template_exact_accuracy
                if evaluation_metric == "exact_accuracy"
                else prompt_template_existence_accuracy
            ),
            ice_template=prompt_template_ICE,
        )

        outfile_name = experiment_dict_to_fname(experiment, extension="csv")
        df_prompt[save_cols].to_csv(os.path.join(output_dir, outfile_name), index=False)


if __name__ == "__main__":
    # Command-line argument parser setup
    parser = argparse.ArgumentParser(
        description="Run the main function with command line arguments."
    )

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
    parser.add_argument(
        "--evaluation_metric",
        type=str,
        choices=["exact_accuracy", "existence_accuracy"],
        required=True,
        help="The evaluation metric to use (either 'exact_accuracy' or 'existence_accuracy').",
    )

    # Parse and run
    args = parser.parse_args()
    args.max_seq_len = None if not args.max_seq_len else args.max_seq_len

    main(
        args.experiments_file,
        args.output_dir,
        args.model_name,
        args.max_seq_len,
        args.test_size,
        args.evaluation_metric,
    )
