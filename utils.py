import json
import pandas as pd
from tqdm import tqdm

dashed_line = '- '*50

def experiment_dict_to_fname(experiment_dict):
    return f"""
    context_lang:{experiment_dict['context_lang']},question_lang:{experiment_dict['question_lang']},noise_type:{experiment_dict['noise_type']},retrieval_type:{experiment_dict['retrieval_type']},needle_position:{experiment_dict['needle_position']}
    """.strip()

def fname_to_experiment_dict(fname):
    experiment_dict = dict()
    fname = fname.split(".")[0]  # Remove file extensions like .txt / .json
    items = fname.split(",")
    for item in items:
        key = item.split(":")[0]
        value = item.split(":")[1]
        experiment_dict[key] = value

    return experiment_dict


def merge_dicts(dict_list):
    merged_dict = {}
    for dictionary in dict_list:
        merged_dict.update(dictionary)
    return merged_dict


def get_top_n_keys(input_dict, n, threshold):
    # Filter the dictionary to remove items below the threshold
    filtered_dict = {
        key: value for key, value in input_dict.items() if value >= threshold
    }

    # Sort the filtered dictionary by value in descending order
    sorted_items = sorted(filtered_dict.items(), key=lambda item: item[1], reverse=True)

    # Extract the top n keys from the sorted items
    top_n_keys = [item[0] for item in sorted_items[:n]]

    return top_n_keys


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
    assert needle_position in ['start', 'end', 'middle'], "Invalid needle_position"

    if len(prompt) <= max_seq_len:
        return prompt  # No truncation needed

    if needle_position == 'start':
        return prompt[-max_seq_len:]  # Truncate from the end
    elif needle_position == 'end':
        return prompt[:max_seq_len]  # Truncate from the start
    elif needle_position == 'middle':
        excess_len = len(prompt) - max_seq_len
        start_len = excess_len // 2
        end_len = excess_len - start_len
        return prompt[start_len:len(prompt)-end_len] # Truncate from both sides