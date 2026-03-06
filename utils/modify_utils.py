import torch
import logging
import os
import random

from PIL import Image
from diffusers import QwenImageEditPipeline
from datetime import datetime

class ModifyUtils:

    device = None
    model = None
    pipeline = None
    processor = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Device: {self.device}")
        self.model = "Qwen/Qwen-Image-Edit"
        self.dtype = torch.bfloat16
        self.pipeline = QwenImageEditPipeline.from_pretrained(
            self.model,
            torch_dtype=self.dtype
        ).to(self.device)
        self.pipeline.vae.enable_slicing()
        self.pipeline.vae.enable_tiling()
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.set_progress_bar_config(disable=None)

    def modify_image(self, image_path, prompt, output_path=None, width=512, height=512):
        image = Image.open(image_path).convert("RGB")
        inputs = {
            "image": image,
            "prompt": prompt,
            "generator": torch.manual_seed(random.randint(0, 1000)),
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "num_inference_steps": 50,
            "width": width,
            "height": height,
        }
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_path = "output" + os.path.sep + timestamp + "_image.png"

        with torch.inference_mode():
            output = self.pipeline(**inputs)
            output_image = output.images[0]
            output_image.save(output_path)
            logging.info(f"Image saved to {output_path}")

        return output_path
