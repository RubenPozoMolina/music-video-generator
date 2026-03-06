import gc
import logging

import torch
from diffusers import WanImageToVideoPipeline
from diffusers.utils import load_image, export_to_video
from moviepy import VideoFileClip, concatenate_videoclips

class VideoUtils:

    device = None
    pipe = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
            torch_dtype=torch.float16
        ).to(self.device)
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        self.pipe.enable_model_cpu_offload()

    @staticmethod
    def free_memory():
        torch.cuda.empty_cache()
        gc.collect()

    def generate_video_from_image_to_image(self, first_image, last_image, prompt, output_path):
        first_frame = load_image(first_image)
        last_frame = load_image(last_image)

        output = self.pipe(
            image=first_frame,
            last_image=last_frame,
            prompt=prompt,
            num_frames=81,
            guidance_scale=6.0,
            num_inference_steps=30
        )

        export_to_video(output.frames[0], output_path, fps=16)
        self.free_memory()
        return output_path


    @staticmethod
    def concatenate_videos(input_videos, output_video):
        clips = []
        try:
            for video in input_videos:
                clip = VideoFileClip(video)
                clips.append(clip)

            final_video = concatenate_videoclips(clips, method="compose")
            final_video.write_videofile(output_video, codec="libx264")
        except Exception as e:
            logging.error(f"Error concatenating clips: {e}")
        finally:
            for clip in clips:
                clip.close()
