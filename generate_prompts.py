import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from prompt_templates import prompt_template1


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

def get_distractor_passages(
    noise_type, matches, num_docs=3, similarity_threshold=0.5
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


def truncate_prompt(prompt: str, max_seq_len: int, needle_position: str) -> str:
    """
    Truncates the input prompt based on the specified needle position and maximum token length.

    Args:
        prompt (str): The input prompt.
        max_token_len (int): The maximum allowed token length.
        needle_position (str): The position to truncate from ('start', 'end', or 'middle').

    Returns:
        str: The truncated prompt.

    Raises:
        AssertionError: If needle_position is not one of ['start', 'end', 'middle'].
    """
    assert needle_position in ["start", "end", "middle"], "Invalid needle_position"

    if len(prompt) <= max_seq_len:
        return prompt  # No truncation needed

    if needle_position == "start":
        return prompt[-max_seq_len:]  # Truncate from the end
    elif needle_position == "end":
        return prompt[:max_seq_len]  # Truncate from the start
    elif needle_position == "middle":
        excess_len = len(prompt) - max_seq_len
        start_len = excess_len // 2
        end_len = excess_len - start_len
        return prompt[start_len : len(prompt) - end_len]  # Truncate from both sides


def generate_prompts(
    data: pd.DataFrame, experiment_dict: dict, prompt_template
) -> list[str]:
    """
    Given an experiment, this code returns a list of prompts for LLM inference
    """
    context_lang = experiment_dict["context_lang"]
    question_lang = experiment_dict["question_lang"]
    noise_type = experiment_dict["noise_type"]
    retrieval_type = experiment_dict["retrieval_type"]
    needle_position = experiment_dict["needle_position"]
    output = []

    df = data.copy()
    df = df[
        (df["context_lang"] == context_lang) & (df["question_lang"] == question_lang)
    ]  # Filter relevant instances
    assert df.shape[1] > 0
    for i, row in tqdm(df.iterrows(), desc="Generating prompts"):

        # Noise retrieval code - retrieve passages from data to be appended to the prompt along with the needle.
        context_para = row["context"]
        noise_passages = retrieve_noise_passages(
            data, context_para, context_lang, retrieval_type, noise_type
        )
        context = insert_needle(
            needle=context_para,
            distractor_passages=noise_passages,
            position=needle_position,
        )
        context = " ".join([passage for passage in context])
        context = context.strip()

        instruction = "Answer the given question correctly based on the given context."
        question = row["question"]
        prompt = prompt_template(instruction, question, context)
        output.append(prompt)

    return output


def retrieve_noise_passages(
    data: pd.DataFrame,
    context_para,
    context_lang,
    retrieval_type,
    noise_type,
    num_paras=1000,
):
    assert retrieval_type in ["related", "unrelated"]
    noise_passages = []

    if retrieval_type == "unrelated":

        if noise_type == "multilingual":
            noise_passages = (
                data[data["context_lang"] != context_lang]
                .sample(num_paras)["context"]
                .values.tolist()
            )

        else:
            noise_passages = (
                data[data["context_lang"] == noise_type]
                .sample(num_paras)["context"]
                .values.tolist()
            )

    return noise_passages

"""
Generate prompts.csv based on the given experiment. 

Sample usage
python generate_prompts.py --experiment_file='' output_dir=''
"""

data = pd.read_csv('/home/amey/long-context-llms/mlqa_combined_formatted.csv')

experiment = {
    'context_lang': 'en', 
    'question_lang': 'hi', 
    'noise_type': 'hi',
    'retrieval_type': 'unrelated', 
    'needle_position': 'middle'
}

prompts = generate_prompts(
    data=data, experiment_dict=experiment
)

