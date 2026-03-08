import logging
import time

from utils.image_utils import ImageUtils

models = [
    "Lykon/DreamShaper",
    "stabilityai/stable-diffusion-3-medium-diffusers",
    "black-forest-labs/FLUX.1-dev"
]


def create_image(model, prompt):
    start_time = time.perf_counter()
    image_utils = ImageUtils(model)
    image_utils.create_image(prompt=prompt)
    del image_utils
    elapsed_time = time.perf_counter() - start_time
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f"Create image completed in {hours}h {minutes}m {seconds}s")


def main():
    logging.basicConfig(level=logging.INFO)
    prompt = "A zombie playing an electric guitar in a dark cemetery under the moonlight"
    for model in models:
        create_image(model, prompt)


if __name__ == "__main__":
    main()
