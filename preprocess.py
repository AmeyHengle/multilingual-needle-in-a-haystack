from typing import Any, Dict, List, Tuple

import torch
from accelerate import Accelerator
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig, LlamaForCausalLM

from utils import dashed_line, remove_substring


class Preprocess:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        from_local: bool = False,
        max_seq_len: int = None,
        buffer: int = 64,
    ):
        self.model_name = model_name

        # Initialize the appropriate tokenizer based on the model name
        if "mistral" in self.model_name.lower():
            self.tokenizer = MistralTokenizer.v2()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.max_seq_len = (
            self.tokenizer.model_max_length if max_seq_len is None else int(max_seq_len)
        )
        self.buffer = buffer

    def get_tokens(self, prompt: str) -> List[int]:
        """Encodes the prompt into tokens based on the model's tokenizer."""
        if "mistral" in self.model_name.lower():
            completion_request = ChatCompletionRequest(
                messages=[UserMessage(content=prompt)]
            )
            tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
            return tokens
        else:
            return self.tokenizer.encode(prompt)

    def truncate_prompt(
        self, prompt: str, needle_position: str, max_token_len: int
    ) -> str:
        """
        Truncates the input prompt based on the specified needle position and maximum token length.

        Args:
            prompt (str): The input prompt.
            needle_position (str): The position to truncate from ('start', 'end', 'middle').
            max_token_len (int): The maximum allowed token length, adjusted by buffer.

        Returns:
            str: The truncated prompt.
        """
        assert needle_position in ["start", "end", "middle"], "Invalid needle_position"

        # Adjust the maximum token length by subtracting the buffer
        if max_token_len > self.buffer:
            max_token_len -= self.buffer

        input_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)

        # Return the original prompt if no truncation is necessary
        if len(input_tokens) <= max_token_len:
            return prompt

        # Truncate based on the specified needle position
        if needle_position == "end":
            input_tokens = input_tokens[-max_token_len:]
        elif needle_position == "start":
            input_tokens = input_tokens[:max_token_len]
        elif needle_position == "middle":
            excess_len = len(input_tokens) - max_token_len
            start_len = excess_len // 2
            end_len = excess_len - start_len
            input_tokens = input_tokens[start_len : len(prompt) - end_len]

        # Decode the truncated tokens back into a string
        truncated_prompt = self.tokenizer.decode(
            input_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return truncated_prompt

    def run(self, prompt_template, prompt: str, needle_position: str) -> str:
        """
        Truncate the prompt to fit within the maximum sequence length after accounting for the template.

        Args:
            prompt_template (function): A function that generates the base prompt.
            prompt (str): The main content of the prompt.
            needle_position (str): The position to truncate from ('start', 'end', or 'middle').

        Returns:
            str: The truncated prompt.
        """
        base_prompt = prompt_template(question="", context="")
        input_tokens = self.tokenizer.tokenize(base_prompt)
        base_prompt_len = len(input_tokens)

        # Adjust the maximum sequence length based on the base prompt length
        if base_prompt_len < self.max_seq_len:
            print(f"Original max sequence length: {self.max_seq_len}")
            adjusted_max_seq_len = self.max_seq_len - base_prompt_len
            print(
                f"Adjusted max sequence length based on prompt template: {adjusted_max_seq_len}"
            )
            print(dashed_line)

        # Truncate the prompt based on the adjusted maximum sequence length
        prompt = self.truncate_prompt(
            prompt,
            needle_position=needle_position,
            max_token_len=adjusted_max_seq_len,
        )

        return prompt

    def preprocess(
        self, prompt_template, needle_passage: str, distractor_passages: List[str]
    ) -> List[str]:
        """
        Truncates the distractor passages to fit within the maximum sequence length.

        Args:
            prompt_template (function): A function that generates the base prompt.
            needle_passage (str): The main content passage.
            distractor_passages (List[str]): A list of distractor passages.

        Returns:
            List[str]: A list of truncated distractor passages.
        """
        distractor_passages_truncated = []
        base_prompt = prompt_template(question="", context="")
        base_tokens = self.get_tokens(base_prompt)
        base_prompt_len = len(base_tokens)

        needle_tokens = self.get_tokens(needle_passage)
        needle_len = len(needle_tokens)

        base_total_len = base_prompt_len + needle_len + self.buffer

        print(f"Original max sequence length: {self.max_seq_len}")
        adjusted_max_seq_len = self.max_seq_len - base_total_len
        print(
            f"Adjusted max sequence length based on prompt template: {adjusted_max_seq_len}"
        )
        print(dashed_line)

        for passage in distractor_passages:
            input_tokens = self.get_tokens(passage)
            if len(input_tokens) < adjusted_max_seq_len:
                distractor_passages_truncated.append(passage)
                adjusted_max_seq_len -= len(input_tokens)
            else:
                input_tokens = input_tokens[:adjusted_max_seq_len]
                truncated_passage = self.tokenizer.decode(input_tokens)
                distractor_passages_truncated.append(truncated_passage)
                break

        return distractor_passages_truncated

    def postprocess(self, prompts: List[str], predictions: List[str]) -> List[str]:
        """
        Cleans up the predictions by removing the corresponding prompt text.

        Args:
            prompts (List[str]): The list of original prompts.
            predictions (List[str]): The list of model predictions.

        Returns:
            List[str]: The cleaned predictions.
        """
        assert len(prompts) == len(predictions)
        predictions_cleaned = []

        for i in tqdm(range(len(prompts)), desc="Post processing prompts."):
            cleaned_str = remove_substring(string=predictions[i], substring=prompts[i])
            cleaned_str = cleaned_str.strip()
            predictions_cleaned.append(cleaned_str)

        return predictions_cleaned
