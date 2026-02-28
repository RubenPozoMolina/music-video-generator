import torch
from transformers import pipeline


class TextUtils:

    pipe = None

    def __init__(self):
        self.pipe = pipeline(
            "image-text-to-text",
            model="google/gemma-3-27b-it",
            device="cuda",
            torch_dtype=torch.bfloat16
        )

    def generate_text(self, messages):
        output = self.pipe(text=messages, max_new_tokens=2000)
        return output[0]["generated_text"][-1]["content"]