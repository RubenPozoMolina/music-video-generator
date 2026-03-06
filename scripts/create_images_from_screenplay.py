import json
import logging
from pathlib import Path

from utils.image_utils import ImageUtils
from utils.modify_utils import ModifyUtils
from utils.video_utils_old import VideoUtils


def main():
    logging.basicConfig(level=logging.INFO)
    with open("data/examples/screenplay.json", "r") as f:
        screenplay = json.load(f)

    height = 480
    width = 832
    output_path = "output/screenplay"
    if not Path(output_path).exists():
        Path(output_path).mkdir(parents=True, exist_ok=True)

    # Create images for each scene
    image_utils = ImageUtils()
    first_image = image_utils.create_image(
        "Lykon/DreamShaper",
        screenplay[0]["description"],
        height,
        width,
        output_path + f"/{screenplay[0]['scene']}.png"
    )
    logging.info(f"First image saved to {first_image}")

    first_image = "output/screenplay/1.png"

    modify_utils = ModifyUtils()
    from_image = first_image
    for scene in screenplay:
        from_image = modify_utils.modify_image(
            from_image,
            scene["description"],
            output_path + f"/{scene['scene']}.png",
            width,
            height
        )
        logging.info(f"Scene {scene['scene']} saved to {from_image}")
        scene["image"] = from_image


if __name__=="__main__":
    main()
