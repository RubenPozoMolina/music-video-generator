import gc
import logging
import torch
from diffusers import WanImageToVideoPipeline
from diffusers.utils import load_image, export_to_video
from moviepy import VideoFileClip, concatenate_videoclips
from PIL import Image


class VideoUtils:

    device = None
    pipe = None

    TARGET_WIDTH = 1280
    TARGET_HEIGHT = 720

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
            torch_dtype=torch.bfloat16
        ).to(self.device)
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        self.pipe.enable_model_cpu_offload()

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
    ) -> str:

        first_frame = self._prepare_frame(
            first_image, self.TARGET_WIDTH, self.TARGET_HEIGHT
        )
        last_frame = self._prepare_frame(
            last_image, self.TARGET_WIDTH, self.TARGET_HEIGHT
        )

        negative_prompt = (
            "blurry, low quality, distortion, artifacts, flickering, "
            "incoherent motion, watermark, text"
        )

        with torch.inference_mode():
            output = self.pipe(
                image=first_frame,
                last_image=last_frame,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=81,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(device=self.device).manual_seed(42),
            )

        export_to_video(output.frames[0], output_path, fps=fps)
        self.free_memory()
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