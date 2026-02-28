import logging
import os.path
from datetime import datetime

import torch
from PIL import Image
from diffusers import FluxPipeline, DiffusionPipeline, QwenImageEditPipeline


class ImageUtils:

    device = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Device: {self.device}")

    def create_image(
            self,
            model,
            prompt,
            height=1024,
            width=1024,
            output_path=None
    ):
        if model is None:
            raise ValueError("Model cannot be None")
        elif model != "FLUX.1-dev":
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.bfloat16
            )
        else:
            pipe = DiffusionPipeline.from_pretrained(
                model,
                torch_dtype=torch.bfloat16
            )
        pipe.to(self.device)
        image = pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator().manual_seed(0)
        ).images[0]
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_path = "output" + os.path.sep +  timestamp + "_image.png"
        image.save(output_path)
        logging.info(f"Image saved to {output_path}")
        return output_path

    def modify_image(self, image_path, prompt, output_path=None):
        pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
        pipeline.to(torch.bfloat16)
        pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=None)
        image = Image.open(image_path).convert("RGB")
        inputs = {
            "image": image,
            "prompt": prompt,
            "generator": torch.manual_seed(0),
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "num_inference_steps": 50,
        }
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_path = "output" + os.path.sep +  timestamp + "_image.png"

        with torch.inference_mode():
            output = pipeline(**inputs)
            output_image = output.images[0]
            output_image.save(output_path)
            logging.info(f"Image saved to {output_path}")
            return output_path