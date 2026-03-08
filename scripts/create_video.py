import argparse
import json
import logging
import pathlib
import time

from utils.gpu_utils import GPUUtils
from utils.image_utils import ImageUtils
from utils.modify_utils import ModifyUtils
from utils.video_utils import VideoUtils


def create_video(config_file, output_path):
    start_time = time.perf_counter()
    logging.info("Loading config file:%s", config_file)
    with open(config_file, "r") as f:
        config = json.load(f)

    title = "No title"
    if "title" in config:
        title = config["title"]
    logging.info(title)

    width = 1280
    if "width" in config:
        width = config["width"]

    height = 720
    if "height" in config:
        height = config["height"]

    main_prompt = ""
    if "prompt" in config:
        main_prompt = config["prompt"]

    logging.info("Main prompt: %s", main_prompt)

    main_negative_prompt = ""
    if "negative_prompt" in config:
        main_negative_prompt = config["negative_prompt"]

    logging.info("Main negative prompt: %s", main_negative_prompt)

    screenplay = []
    if "screenplay" in config:
        screenplay = config["screenplay"]

    # Create the output path if not exists
    if pathlib.Path(output_path).exists():
        logging.info("Output path exists: %s", output_path)
    else:
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        logging.info("Created output path: %s", output_path)

    if len(screenplay) == 0:
        logging.error("Screenplay is empty")
        return

    # Create the first image
    logging.info("Creating first image")
    first_scene = screenplay[0]
    first_image_path = output_path + "/1.png"
    if pathlib.Path(first_image_path).exists():
        logging.info("First image exists: %s", first_image_path)
    else:
        prompt = main_prompt
        prompt += first_scene["prompt"]
        image_utils = ImageUtils("black-forest-labs/FLUX.1-dev")
        image_utils.create_image(
            prompt=prompt,
            width=width,
            height=height,
            output_path=first_image_path
        )

    GPUUtils.free_memory()

    # Create images from the first image
    logging.info("Creating images from the first image")
    from_image = first_image_path
    modify_utils = ModifyUtils()
    for scene in screenplay:
        to_image = output_path + f"/{scene['scene']}.png"
        if pathlib.Path(to_image).exists():
            logging.info("Image exists: %s", to_image)
            continue
        prompt = main_prompt
        prompt += scene["prompt"]
        from_image = modify_utils.modify_image(
            from_image,
            prompt=prompt,
            negative_prompt=main_negative_prompt,
            output_path=output_path + f"/{scene['scene']}.png",
            width=width,
            height=height
        )
        logging.info(f"Scene {scene['scene']} saved to {from_image}")

    GPUUtils.free_memory()

    # Create videos from images
    logging.info("Creating videos from images")
    video_utils = VideoUtils("Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers")
    clips = []
    from_image = first_image_path
    for i in range(1, len(screenplay)):
        scene = screenplay[i]
        to_image = f"{output_path}/{scene['scene']}.png"
        video_output = output_path + f"/{scene['scene']}.mp4"
        if pathlib.Path(video_output).exists():
            logging.info("Video exists: %s", video_output)
            continue
        video_utils.generate_video_from_image_to_image(
            from_image,
            to_image,
            scene["prompt"],
            video_output,
            50
        )
        logging.info(f"Scene {scene['scene']} saved to {scene['scene']}.mp4")
        from_image = to_image
        scene["video"] = output_path + f"/{scene['scene']}.mp4"
        clips.append(scene["video"])

    video_utils.concatenate_videos(clips, output_path + "/final_video.mp4")
    logging.info(f"Final video saved to {output_path}/final_video.mp4")

    elapsed_time = time.perf_counter() - start_time
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f"create_video completed in {hours}h {minutes}m {seconds}s")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Create a music video'
    )
    parser.add_argument(
        '--config-file',
        dest='config_file',
        help='config file to create the music video'
    )
    parser.add_argument(
        '--output-path',
        dest='output_path',
        help='folder to generate output files'
    )
    parsed = parser.parse_args()

    create_video(parsed.config_file, parsed.output_path)
