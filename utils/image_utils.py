import gc
import logging
import os.path
import random
import time
from datetime import datetime

import torch
from diffusers import FluxPipeline, DiffusionPipeline, StableDiffusion3Pipeline

from utils.gpu_utils import GPUUtils
from utils.logging_utils import show_elapsed_time


class ImageUtils:

    device = None
    pipeline = None
    model = None


    def __init__(self, model: str = "Lykon/DreamShaper"):
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Device: {self.device}")
        self.model = model

    def _load_pipeline(self):
        start_time = time.perf_counter()
        if self.model.startswith("black-forest-labs"):
            self.pipeline = FluxPipeline.from_pretrained(
                self.model,
                torch_dtype=torch.bfloat16
            )
        elif self.model.startswith("stabilityai"):
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                self.model,
                torch_dtype=torch.bfloat16
            )
        else:
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.model,
                torch_dtype=torch.bfloat16
            )
        self.pipeline.to(self.device)
        logging.info(f"Pipeline loaded for model: {self.model}")
        GPUUtils.show_mem()
        show_elapsed_time(start_time)

    def create_image(
            self,
            prompt,
            negative_prompt="",
            height=720,
            width=1280,
            output_path=None,
            seed = None
    ):
        if self.pipeline is None:
            self._load_pipeline()
        start_time = time.perf_counter()
        effective_seed = seed if seed is not None else random.randint(0, 1000000)
        generator = torch.Generator(device=self.device).manual_seed(effective_seed)
        image = self.pipeline(
            prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=generator
        ).images[0]
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_path = "output" + os.path.sep +  timestamp + "_image.png"
        image.save(output_path)
        logging.info(f"Image saved to {output_path}")
        GPUUtils.show_mem()
        show_elapsed_time(start_time)
        return output_path

    def unload(self):
        self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logging.info("Unload memory")
        GPUUtils.show_mem()

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