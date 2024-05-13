import torch
from typing import Any, Dict, List, Tuple, cast
from utils import dashed_line
from transformers import T5Tokenizer, T5ForConditionalGeneration
from accelerate import Accelerator
from tqdm import tqdm

class FlanT5:
    def __init__(
        self,
        t5_model_name: str = "google/flan-t5-small",
        from_local: bool = False,
        max_seq_len: int = None,
        torch_dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        assert (
            "t5" in t5_model_name or from_local
        ), "Explain model only supports T5 models."
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(t5_model_name).to(self.device)
            self.max_seq_len = self.tokenizer.model_max_length if max_seq_len == None else int(max_seq_len)
        except Exception as e:
            print(f"Error loading model {t5_model_name}: {e}")
            raise e

    def truncate_prompt(self, prompt: str, needle_position: str) -> str:
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
        
        input_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        if len(input_tokens) <= self.max_seq_len:
            return prompt  # No truncation needed

        if needle_position == 'start':
            # Truncate from the end
            input_tokens = input_tokens[-self.max_seq_len:]
        elif needle_position == 'end':
            # Truncate from the start
            input_tokens =  input_tokens[:self.max_seq_len]  
        elif needle_position == 'middle':
            excess_len = len(input_tokens) - self.max_seq_len
            start_len = excess_len // 2
            end_len = excess_len - start_len
            input_tokens =  input_tokens[start_len:len(prompt)-end_len] # Truncate from both sides

        truncated_prompt = self.tokenizer.decode(input_tokens)
        return truncated_prompt

    def preprocess(self, prompt_template, prompts:list[str])->list[str]:
        # Truncate prompts based so as to fit max_seq_len. 
        # The code ensures that only "in context examples" are truncated. 
        # The main parts of the prompts, such as the instruction, question, etc remain intact. 

        base_prompt = prompt_template(question="", context="")
        input_tokens = self.tokenizer.tokenize(base_prompt)
        base_prompt_len = len(input_tokens)
        if base_prompt_len < self.max_seq_len:
            print(f"Original max sequence length: {self.max_seq_len}")
            self.max_seq_len = self.max_seq_len - base_prompt_len
            print(f"Adjusted max sequence length based on prompt template: {self.max_seq_len}")
            print(dashed_line)
            
        for i in tqdm(range(len(prompts)), desc="Preprocessing Prompts"):
            prompts[i] = self.truncate_prompt(prompts[i])

        return prompts
            
    
    def predict(self, prompt_template, prompts:list[str]):
        # Preprocess prompts to follow max_seq_len
        prompts = self.preprocess(prompt_template, prompts)

        # Initialize the accelerator
        accelerator = Accelerator()
        
        # Use accelerator.prepare to make sure everything is on the right device
        self.model = accelerator.prepare(self.model)

        inputs = self.tokenizer(prompts, return_tensors="pt", truncation=True, max_length=self.max_seq_len)

        # Use the accelerator's device placement for the inputs
        inputs = accelerator.prepare(inputs).to(accelerator.device)

        with accelerator.autocast():
            outputs = self.model.generate(**inputs)

        predictions = [self.tokenizer.decode(output[0], skip_special_tokens=True) for output in outputs]

        return predictions
