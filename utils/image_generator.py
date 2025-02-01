import datetime
import os
import torch

from diffusers import StableDiffusionPipeline


class ImageGenerator:

    prompt = None
    model = None
    width = None
    height = None
    _pipe = None

    def __init__(
            self,
            prompt,
            model="CompVis/stable-diffusion-v1-4",
            width=512, height=512
    ):
        self.prompt = prompt
        self.model = model
        self.width = width
        self.height = height
        self._pipe = StableDiffusionPipeline.from_pretrained(
            self.model, torch_dtype=torch.float16
        )
        self._pipe = self._pipe.to("cuda")

    @staticmethod
    def create_directory_if_not_exists(directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def generate(self, output_path, file_name = None):
        image = self._pipe(
            self.prompt, height=self.height, width=self.width
        ).images[0]
        if not file_name:
            file_name = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        file_path = f"{output_path}/{file_name}.png"
        self.create_directory_if_not_exists(output_path)
        image.save(file_path)
        return image