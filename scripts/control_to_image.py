import datetime
import random
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

pipe.load_lora_weights("lora/KawaiiStyle_Style-20.safetensors")
pose_image_1 = load_image("input/control_1.png")
pose_image_2 = load_image("input/control_2.png")

prompt = "A unique cute gray cat dancing on the beach, less than 5 legs, less than 2 tails"
negative_prompt = "human face, human hand, human legs, human body, human"

generator = torch.manual_seed(random.randint(0, 1000))

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=pose_image_1,
    num_inference_steps=50,
    guidance_scale=7.5,
    strength=0.9,
    controlnet_conditioning_scale=1.2,
    generator=generator
).images[0]

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
output_file = "output/" + timestamp + ".png"
image.save(output_file)

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=pose_image_2,
    num_inference_steps=50,
    guidance_scale=7.5,
    strength=0.9,
    controlnet_conditioning_scale=1.2,
    generator=generator
).images[0]

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
output_file = "output/" + timestamp + ".png"
image.save(output_file)