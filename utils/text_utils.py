import json
import logging
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig


class TextUtils:
    pipe = None
    device = None
    model = None

    def __init__(self, model: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    @staticmethod
    def save_dict_to_file(data, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    @staticmethod
    def _clean_output(raw_text):
        pattern = r"```json\s*(.*?)\s*```"
        match = re.search(pattern, raw_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return raw_text.strip()

    def generate_dict_list(self, prompt, schema_description):
        full_prompt = (
            f"Act as json generator. {prompt}. "
            f"Mandatory create a list following this schema: {schema_description}. "
            "Answer only in JSON format list. Without any explanation."
        )

        messages = [
            {"role": "system", "content": "You are an expert in JSON generation."},
            {"role": "user", "content": full_prompt}
        ]

        outputs = self.pipe(messages, max_new_tokens=512, return_full_text=False)
        raw_response = outputs[0]['generated_text']

        clean_json_str = self._clean_output(raw_response)

        try:
            return json.loads(clean_json_str)
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON: {clean_json_str}")
            return {"raw_output": raw_response}

