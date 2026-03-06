import gc
import logging
import os.path
from datetime import datetime

import torch
from diffusers import FluxPipeline, DiffusionPipeline


class ImageUtils:

    device = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Device: {self.device}")
        self.pipeline = None
        self.processor = None

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
        elif model == "FLUX.1-dev":
            self.pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.bfloat16
            )
        else:
            self.pipeline = DiffusionPipeline.from_pretrained(
                model,
                torch_dtype=torch.bfloat16
            )
        self.pipeline.to(self.device)
        if self.pipeline:
            image = self.pipeline(
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

    def unload(self):
        self.pipeline = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def close(self):
        self.unload()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.unload()
        return False

    def __del__(self):
        try:
            self.unload()
        except Exception as e:
            logging.warning(f"Error freeing memory: {e}")
            pass