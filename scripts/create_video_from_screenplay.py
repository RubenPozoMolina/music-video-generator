import json
import logging

from utils.video_utils_old import VideoUtils


def main():
    logging.basicConfig(level=logging.INFO)
    input_path = "data/examples"
    with open(f"{input_path}/screenplay.json", "r") as f:
        screenplay = json.load(f)

    output_path = "output/screenplay"

    video_utils = VideoUtils()
    clips = []
    from_image = f"{input_path}/1.png"
    for i in range(1, len(screenplay)):
        scene = screenplay[i]
        to_image = f"{input_path}/{scene['scene']}.png"
        video_utils.generate_video_from_image_to_image(
            from_image,
            to_image,
            scene["description"],
            output_path + f"/{scene['scene']}.mp4"
        )
        logging.info(f"Scene {scene['scene']} saved to {scene['scene']}.mp4")
        from_image = to_image
        scene["video"] = output_path + f"/{scene['scene']}.mp4"
        clips.append(scene["video"])

    video_utils.concatenate_videos(clips, output_path + "/final_video.mp4")
    logging.info(f"Final video saved to {output_path}/final_video.mp4")


if __name__=="__main__":
    main()
