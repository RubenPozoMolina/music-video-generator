from PIL import Image
import torch

from diffusers import QwenImageEditPipeline

pipeline = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    torch_dtype=torch.bfloat16,
).to("cuda")
print("pipeline loaded")
pipeline.enable_model_cpu_offload()
pipeline.set_progress_bar_config(disable=None)
image = Image.open("data/examples/zombie.png").convert("RGB")
prompt = "The zombie raise the guitar to the moon"
inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_file = "output/output_image.png"
    output_image.save(output_file)
    print("image saved at", output_file)
