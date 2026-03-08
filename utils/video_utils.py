import gc
import logging
import random
import subprocess
import time

import torch
from diffusers import WanImageToVideoPipeline, StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from moviepy import VideoFileClip, concatenate_videoclips
from PIL import Image

from utils.gpu_utils import GPUUtils
from utils.logging_utils import show_elapsed_time


class VideoUtils:

    device = None
    model = None
    pipeline = None


    def __init__(self, model):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        GPUUtils.show_gpu_info()

    def _load_pipeline(self):
        start_time = time.perf_counter()
        if "Wan" in self.model:
            self.pipeline = WanImageToVideoPipeline.from_pretrained(
                self.model,
                torch_dtype=torch.bfloat16
            )
            self.pipeline.to(self.device)
        elif "stability" in self.model:
            self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                self.model,
                torch_dtype=torch.float16,
                variant="fp16"
            )
            self.pipeline.to(self.device)
        else:
            raise Exception("Model not loaded")
        logging.info("Loaded model %s", self.model)
        GPUUtils.show_mem()
        show_elapsed_time(start_time)

    @staticmethod
    def free_memory():
        torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def _prepare_frame(image_path: str, width: int, height: int) -> Image.Image:
        img = load_image(image_path).convert("RGB")
        img_ratio = img.width / img.height
        target_ratio = width / height

        if img_ratio > target_ratio:
            new_w = width
            new_h = round(width / img_ratio)
        else:
            new_h = height
            new_w = round(height * img_ratio)

        img = img.resize((new_w, new_h), Image.LANCZOS)

        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        offset_x = (width - new_w) // 2
        offset_y = (height - new_h) // 2
        canvas.paste(img, (offset_x, offset_y))
        return canvas

    def generate_video_from_image_to_image(
        self,
        first_image: str,
        last_image: str,
        prompt: str,
        output_path: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 8.0,
        fps: int = 24,
        seed: int = None,
        width: int = 1280,
        height: int = 780
    ) -> str:
        if self.pipeline is None:
            self._load_pipeline()
        start_time = time.perf_counter()
        effective_seed = seed if seed is not None else random.randint(0, 1000000)
        generator = torch.Generator(device=self.device).manual_seed(effective_seed)
        first_frame = self._prepare_frame(
            first_image, width, height
        )
        last_frame = self._prepare_frame(
            last_image, width, height
        )

        negative_prompt = (
            "blurry, low quality, distortion, artifacts, flickering, "
            "incoherent motion, watermark, text"
        )

        with torch.inference_mode():
            try:
                output = self.pipeline(
                    image=first_frame,
                    last_image=last_frame,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_frames=81,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                )
            except torch.cuda.OutOfMemoryError as e:
                logging.error(f"OOM while inference: {e}")
                self.free_memory()
                raise

        export_to_video(output.frames[0], output_path, fps=fps)
        GPUUtils.show_mem()
        show_elapsed_time(start_time)
        self.free_memory()
        GPUUtils.show_mem()
        return output_path

    @staticmethod
    def concatenate_videos(input_videos: list[str], output_video: str):
        clips = []
        try:
            for video in input_videos:
                clip = VideoFileClip(video)
                clips.append(clip)

            final_video = concatenate_videoclips(clips, method="compose")
            final_video.write_videofile(
                output_video,
                codec="libx264",
                fps=24,
                audio=False,
            )
        except Exception as e:
            logging.error(f"Error concatenating clips: {e}")
        finally:
            for clip in clips:
                clip.close()

    @staticmethod
    def extract_last_frame(video_path, output_path):
        # Use ffmpeg to extract the last frame of the video
        command = [
            "ffmpeg",
            "-i", video_path,
            "-vframes", "1",
            "-ss", "00:00:01",
            output_path
        ]
        subprocess.run(command, check=True)
        logging.info(f"Last frame saved to {output_path}")