from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class ChatBot:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    def generate_response(self, message: str):
        inputs = self.tokenizer.encode(message, return_tensors="pt")

        outputs = self.model.generate(
            inputs,
            max_length=150,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return response
