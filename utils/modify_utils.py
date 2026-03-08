import gc
import time

import torch
import logging
import os
import random

from PIL import Image
from diffusers import QwenImageEditPipeline, Flux2KleinPipeline
from datetime import datetime

from utils.gpu_utils import GPUUtils
from utils.logging_utils import show_elapsed_time


class ModifyUtils:

    device = None
    model = None
    pipeline = None

    def __init__(self, model="Qwen/Qwen-Image-Edit"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Device: {self.device}")
        self.model = model
        self.dtype = torch.bfloat16

    def _load_pipeline(self):
        start_time = time.perf_counter()
        if "Qwen" in self.model:
            self.pipeline = QwenImageEditPipeline.from_pretrained(
                self.model,
                torch_dtype=self.dtype,
            )
            self.pipeline.to(self.device)
        elif "FLUX" in self.model:
            self.pipeline = Flux2KleinPipeline.from_pretrained(
                self.model,
                torch_dtype=self.dtype
            )
            self.pipeline.to(self.device)
        else:
            raise Exception("Model not loaded")
        logging.info("Loaded model %s", self.model)
        GPUUtils.show_mem()
        show_elapsed_time(start_time)

    def modify_image(
            self,
            image_path,
            prompt,
            negative_prompt = "",
            output_path=None,
            width=1280,
            height=720,
            seed=None
    ):
        if self.pipeline is None:
            self._load_pipeline()
        start_time = time.perf_counter()
        effective_seed = seed if seed is not None else random.randint(0, 1000000)
        generator = torch.Generator(device=self.device).manual_seed(effective_seed)
        image = Image.open(image_path).convert("RGB")
        inputs = {
            "image": image,
            "prompt": prompt,
            "num_inference_steps": 40,
            "width": width,
            "height": height,
            "generator": generator
        }
        if "Qwen" in self.model:
            inputs["true_cfg_scale"] = 4.0
            inputs["negative_prompt"] = negative_prompt
        if "FLUX" in self.model:
            inputs["guidance_scale"] = 1.0
            inputs["num_inference_steps"] = 4

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_path = "output" + os.path.sep + timestamp + "_image.png"

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.inference_mode():
            output = self.pipeline(**inputs)
            output_image = output.images[0]
            output_image.save(output_path)
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