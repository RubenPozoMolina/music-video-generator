import argparse
import datetime
import json
import logging
import os

import torch
from controlnet_aux import OpenposeDetector
from diffusers import UniPCMultistepScheduler, \
    StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image

from utils.file_utils import get_all_files, download_file, get_first_file
from utils.video_utils import create_video_from_images, video_to_frames, \
    download_video, get_video_frames


def frames_to_control(frames_path, controls_path):
    if not os.path.exists(controls_path):
        os.makedirs(controls_path)
        files = get_all_files(frames_path)
        processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
        for file in files:
            print(file)
            image = load_image(file)
            # control_image = processor(image, hand_and_face=True)
            control_image = processor(image)
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
            num_inference_steps=50,
            guidance_scale=5,
            strength=0.5,
        ).images[0]
        image.save(images_path + os.sep + file.split("/")[-1])
        source_image = image


def music_video_generator(
    config_file
):
    with open(config_file, "r") as file:
        config = json.load(file)
    output_folder = config["output"]
    create_directory_if_not_exists(output_folder)
    for scene in config["scenes"]:
        scenes_path = output_folder + os.sep + scene["path"]
        create_directory_if_not_exists(scenes_path )
        video_path = scenes_path + os.sep + scene["video_file"]
        if "video_url" in scene:
            if not os.path.exists(video_path):
                download_video(
                    scene["video_url"],
                    video_path
                )
        if "lora_url" in scene:
            if "lora_file" in scene:
                lora_path = scenes_path + os.sep + "lora"
                if not os.path.exists(lora_path):
                    if not os.path.exists(lora_path):
                        os.makedirs(lora_path)
                    download_file(
                        scene["lora_url"],
                        lora_path + os.sep + scene["lora_file"]
                    )
        if os.path.exists(video_path):
            frames_info = get_video_frames(video_path)
            video_to_frames(
                video_path,
                scenes_path + "/frames"
            )
            frames_to_control(
                scenes_path + "/frames",
                scenes_path + "/controls"
            )
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            images_folder = output_folder + "/" + timestamp
            images_folder += "/images"
            lora = None
            if os.path.exists(scene["lora_file"]):
                lora = scene["lora_file"]
            init_image_path = get_first_file(scenes_path + "/frames")
            init_image = load_image(init_image_path)

            controls_to_images(
                scenes_path + "/controls",
                scenes_path + "/images",
                scene["model_id"],
                scene["prompt"],
                init_image,
                scene["negative_prompt"],
                lora
            )
            # create_video_from_images(
            #     images_folder,
            #     images_folder + "/video.mp4"
            # )
        else:
            logging.warning(
                "Video file not found: " + scene["video_file"]
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Music Video Generator")
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Config file for music video generator"
    )
    args = parser.parse_args()

    music_video_generator(
        args.config_file
    )
