import argparse
import datetime
import os

import torch
from controlnet_aux import OpenposeDetector
from diffusers import UniPCMultistepScheduler, \
    StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image

from utils.file_utils import get_all_files
from utils.video_utils import create_video_from_images, video_to_frames


def frames_to_control(frames_path, controls_path):
    if not os.path.exists(controls_path):
        os.makedirs(controls_path)
        files = get_all_files(frames_path)
        processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
        for file in files:
            print(file)
            image = load_image(file)
            control_image = processor(image, hand_and_face=True)
            control_image.save(controls_path + os.sep + file.split("/")[-1])


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def controls_to_images(
        controls_path, images_path, model_id,
        prompt, init_image,
        negative_prompt="", lora=None
):
    create_directory_if_not_exists(images_path)
    files = get_all_files(controls_path)
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose",
        torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id, controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to("cuda")
    torch.cuda.empty_cache()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    if lora:
        lora_file_path = lora
        pipe.load_lora_weights(lora_file_path)
    source_image = load_image(init_image)
    for file in files:
        control_image = load_image(file)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            init_image=source_image,
            image=control_image,
            num_inference_steps=30,
            guidance_scale=20,
            strength=0.7,
        ).images[0]
        image.save(images_path + os.sep + file.split("/")[-1])
        source_image = image


def music_video_generator(
        input_video,
        output_folder,
        model_id,
        prompt, init_image,
        negative_prompt="", lora=None
):
    video_to_frames(
        input_video,
        output_folder + "/frames"
    )
    frames_to_control(
        output_folder + "/frames",
        output_folder + "/controls"
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    images_folder = output_folder + "/" + timestamp
    images_folder += "/images"
    controls_to_images(
        output_folder + "/controls",
        images_folder,
        model_id,
        prompt,
        init_image,
        negative_prompt,
        lora
    )
    create_video_from_images(
        images_folder,
        images_folder + "/video.mp4"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Music Video Generator")
    parser.add_argument(
        "--input-video",
        type=str,
        required=True,
        help="Input video path"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Output folder to save the generated images"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="sd-legacy/stable-diffusion-v1-5",
        help="Model ID for Stable Diffusion"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt for image generation"
    )
    parser.add_argument(
        "--init-image",
        type=str,
        required=True,
        help="First image for image generation"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        required=False,
        help="Negative prompt for image generation"
    )
    parser.add_argument(
        "--lora",
        type=str,
        required=False,
        help="Lora for image generation"
    )

    args = parser.parse_args()

    music_video_generator(
        args.input_video,
        args.output_folder,
        args.model_id,
        args.prompt,
        args.init_image,
        args.negative_prompt,
        args.lora
    )
