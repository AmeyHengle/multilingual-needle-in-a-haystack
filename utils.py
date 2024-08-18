import ast
import json
import os
import re

import pandas as pd
from tqdm import tqdm

dashed_line = "- " * 50


def remove_substring(string, substring):
    return string.replace(substring, "")


def load_json(filepath, is_str=False):
    """Load a JSON file from a filepath."""
    with open(filepath, "r") as f:
        data = json.loads(json.load(f)) if is_str else json.load(f)
    return data


def save_json(data, filepath):
    """Store a dictionary as a JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f)


def extract_correct_json_substring(text):
    # This pattern looks for the last "{" before a "}" and captures until "}"
    pattern = r"\{[^{]*\}"
    match = re.search(pattern, text)
    if match:
        # Extracts the substring that matches the pattern
        return match.group(0)
    else:
        # Return None if no match is found
        return text


def experiment_dict_to_fname(experiment_dict, extension):
    return f"""
context_lang:{experiment_dict['context_lang']},question_lang:{experiment_dict['question_lang']},noise_type:{experiment_dict['noise_type']},retrieval_type:{experiment_dict['retrieval_type']},needle_position:{experiment_dict['needle_position']}.{extension}
    """.strip()


def fname_to_experiment_dict(fname):
    experiment_dict = dict()

    if ":" not in fname:  # Handle fname annomaly
        fname = fname.replace("context_lang_", "context_lang:")
        fname = fname.replace("question_lang_", "question_lang:")
        fname = fname.replace("noise_type_", "context_lang:")
        fname = fname.replace("retrieval_type_", "retrieval_type:")
        fname = fname.replace("needle_position_", "needle_position:")

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


def truncate_prompt(
    self, prompt: str, needle_position: str, max_token_len: int, buffer: int = 16
) -> str:
    """
    Truncates the input prompt based on the specified needle position and maximum token length.

    Args:
        prompt (str): The input prompt.
        max_token_len (int): The maximum allowed token length.
        needle_position (str): The position to truncate from ('start', 'end', or 'middle').
        buffer (int): Buffer for special tokens

    Returns:
        str: The truncated prompt.

    Raises:
        AssertionError: If needle_position is not one of ['start', 'end', 'middle'].
    """
    assert needle_position in ["start", "end", "middle"], "Invalid needle_position"
    if max_token_len > buffer:
        max_token_len -= buffer

    input_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)

    if len(input_tokens) <= max_token_len:
        return prompt  # No truncation needed

    if needle_position == "end":
        # Truncate from the start
        input_tokens = input_tokens[-max_token_len:]
    elif needle_position == "start":
        # Truncate from the end
        input_tokens = input_tokens[:max_token_len]
    elif needle_position == "middle":
        excess_len = len(input_tokens) - max_token_len
        start_len = excess_len // 2
        end_len = excess_len - start_len
        input_tokens = input_tokens[
            start_len : len(prompt) - end_len
        ]  # Truncate from both sides

    truncated_prompt = self.tokenizer.decode(
        input_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return truncated_prompt


import ast
import glob
import json
import re
import string

import pandas as pd

from utils import dashed_line


def clean_str(input_string) -> str:
    if isinstance(input_string, int):
        input_string = str(input_string)
    input_string = input_string.lower()
    input_string = input_string.replace("here is the answer in json format:", "")
    input_string = input_string.strip()  # Remove extra line spaces
    input_string = input_string.replace("\n", " ")  # Replace line spaces with /s<>
    input_string = input_string.replace("\\", " ")
    input_string = input_string.replace("/", " ")
    input_string = input_string.replace("  ", " ")
    input_string = input_string.lower()  # Lower all characters.
    input_string = input_string.replace(",", "")
    return input_string


def postprocess_llm_pred(llm_prediction: str) -> str:
    """
    Function to parse the right answer from noisy LLM prediction
    """
    if (
        not isinstance(llm_prediction, str)
        or len(llm_prediction) < 1
        or llm_prediction == "{}"
    ):  # Return default value in case prediction == None or empty.
        answer = "<None>"
        return answer
    else:
        if (
            "Question:" in llm_prediction
        ):  # For models like llama, output is both the prompt and prediction.
            llm_prediction = llm_prediction.split("Answer:")[-1]
        if isinstance(llm_prediction, int):
            llm_prediction = str(llm_prediction)
        llm_prediction = clean_str(llm_prediction)
        if len(llm_prediction) < 1:
            return "<Empty>"

        if (
            "{" in llm_prediction and "}" in llm_prediction
        ):  # Check if any json / dict object is present within the prediction.
            llm_prediction = llm_prediction.split("}")[0] + "}"
            llm_prediction = extract_correct_json_substring(llm_prediction)
            dict_str = extract_json(llm_prediction)  # Extract dict object.
            if dict_str is not None:  # Succefully extracted dict from string.
                dict_obj = str_to_dict(dict_str)
                if (
                    isinstance(dict_obj, dict) and len(dict_obj) > 0
                ):  # Succesfully converted string to dict
                    try:
                        key = list(dict_obj.keys())[0]
                        key_ = remove_punctuation(key)
                        key_ = key_.strip()
                        key_ = key_.lower()
                        dict_obj[key_] = dict_obj[key]
                    except:
                        print(dict_obj)
                        # print()
                        return remove_punctuation(llm_prediction)

                    if not "answer" in dict_obj:
                        # print(
                        #    f"Encountered a prediction with dict object without 'answer' key:\n{llm_prediction}"
                        # )
                        return remove_punctuation(llm_prediction)

                    answer = dict_obj["answer"]
                    answer = remove_punctuation(answer)
                    return answer.strip()

        else:  # No json / dict object in prediction.
            answer = remove_punctuation(llm_prediction)
            return answer.strip()

    return remove_punctuation(llm_prediction.strip())


def extract_json(input_string):
    start_index = input_string.find("{")
    end_index = input_string.find("}")

    if start_index == -1 or end_index == -1 or end_index <= start_index:
        return None
    else:
        dict_str = "{" + input_string[start_index + 1 : end_index] + "}"
        return dict_str


def str_to_dict(input_string):
    if not isinstance(input_string, str):
        input_string = str(input_string)

    input_string = input_string.strip()
    if len(input_string.split(":")) == 2:  # dict object in string and not a set object
        input_string = input_string.split(":")
        key = input_string[0]
        key = key.lower().strip().replace('"', "").replace("'", "").replace("{", "")
        key = f'"{key}"'

        value = input_string[1]
        value = value.lower().strip().replace('"', "").replace("'", "").replace("}", "")
        value = f'"{value}"'

        input_string = "{" + key + ":" + value + "}"

    try:
        dict_obj = json.loads(input_string)
        if isinstance(dict_obj, set):
            dict_obj = {"answer": list(dict_obj)[0]}
        return dict_obj
    except:
        try:
            dict_obj = ast.literal_eval(input_string)
            if isinstance(dict_obj, set):
                dict_obj = {"answer": list(dict_obj)[0]}
            return dict_obj
        except:
            print(f"Could not convert the following string to dict:\n{input_string}")
            print(dashed_line)
            return None


def remove_punctuation(input_string, replace_by=""):
    if not isinstance(input_string, str):
        input_string = str(input_string)
    punctuations = string.punctuation
    punctuations = punctuations.replace("<", "")
    punctuations = punctuations.replace(">", "")
    input_string = input_string.replace("  ", " ")
    return f"{replace_by}".join(
        char for char in input_string if char not in punctuations
    )


def is_multilingual(prediction: str) -> bool:
    """
    This code inputs a noisy, multingual prediction, and proceeds to return an equivalent English translation of it.
    """
    if detect(prediction) in [
        "mr",
        "hi",
        "ar",
        "zh",
        "de",
        "es",
        "vi",
        "zh-cn",
        "zh-tw",
    ]:
        print(detect(prediction), prediction)
        return True
    words = prediction.split()
    for word in words:
        if detect(word) in ["mr", "hi", "ar", "zh", "de", "es", "vi", "zh-cn", "zh-tw"]:
            print(detect(word), prediction)
            return True
    return False
