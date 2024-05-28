from typing import Any, Dict, List, Tuple, cast

import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig, LlamaForCausalLM

from utils import dashed_line, remove_substring


class Preprocess:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        from_local: bool = False,
        max_seq_len: int = None,
        buffer: int = 128,
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.max_seq_len = (
            self.tokenizer.model_max_length if max_seq_len == None else int(max_seq_len)
        )
        self.buffer = buffer

    def truncate_prompt(
        self, prompt: str, needle_position: str, max_token_len: int
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
        if max_token_len > self.buffer:
            max_token_len -= self.buffer

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

        if "t5" in self.model_name.lower():
            truncated_prompt = self.tokenizer.decode(input_tokens)

        elif "llama" in self.model_name.lower():
            truncated_prompt = self.tokenizer.decode(
                input_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        elif "mistral" in self.model_name.lower():
            truncated_prompt = self.tokenizer.decode(
                input_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        else:
            raise Exception(f"Model not supported: {self.model_name}")

        return truncated_prompt

    def run(self, prompt_template, prompt: str, needle_position: str) -> list[str]:
        # Truncate prompts based so as to fit max_seq_len.
        # The code ensures that only "in context examples" are truncated.
        # The main parts of the prompts, such as the instruction, question, etc remain intact.

        base_prompt = prompt_template(question="", context="")
        input_tokens = self.tokenizer.tokenize(base_prompt)
        base_prompt_len = len(input_tokens)
        if base_prompt_len < self.max_seq_len:
            print(f"Original max sequence length: {self.max_seq_len}")
            adjusted_max_seq_len = self.max_seq_len - base_prompt_len
            print(
                f"Adjusted max sequence length based on prompt template: {adjusted_max_seq_len}"
            )
            print(dashed_line)

        prompt = self.truncate_prompt(
            prompt,
            needle_position=needle_position,
            max_token_len=adjusted_max_seq_len,
        )

        return prompt

    def preprocess(
        self, prompt_template, needle_passage: str, distractor_passages: list[str]
    ) -> list[str]:
        # Truncate prompts based so as to fit max_seq_len.
        # The code ensures that only "in context examples" are truncated.
        # The main parts of the prompts, such as the instruction, question, etc remain intact.
        distractor_passages_truncated = []
        base_prompt = prompt_template(question="", context="")
        base_tokens = self.tokenizer.tokenize(base_prompt)
        base_prompt_len = len(base_tokens)

        needle_tokens = self.tokenizer.tokenize(needle_passage)
        needle_len = len(needle_tokens)

        base_total_len = base_prompt_len + needle_len + self.buffer

        # if base_total_len < self.max_seq_len:
        print(f"Original max sequence length: {self.max_seq_len}")
        adjusted_max_seq_len = self.max_seq_len - base_total_len
        print(
            f"Adjusted max sequence length based on prompt template: {adjusted_max_seq_len}"
        )
        print(dashed_line)

        for passage in distractor_passages:
            input_tokens = self.tokenizer.tokenize(passage)
            if len(input_tokens) < adjusted_max_seq_len:
                distractor_passages_truncated.append(passage)
                adjusted_max_seq_len -= len(input_tokens)
            else:
                distractor_passages_truncated.append(
                    passage[: adjusted_max_seq_len - len(input_tokens)]
                )
                break

        return distractor_passages_truncated

    def postprocess(self, prompts: list[str], predictions: list[str]) -> list[str]:
        # Post process predictions based on some logic.
        assert len(prompts) == len(predictions)
        predictions_cleaned = []

        for i in tqdm(range(len(prompts)), desc="Post processing prompts."):
            cleaned_str = remove_substring(string=predictions[i], substring=prompts[i])
            cleaned_str = cleaned_str.strip()
            predictions_cleaned.append(cleaned_str)

        return predictions_cleaned
