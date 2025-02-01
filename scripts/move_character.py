import datetime
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image

model = "lykon/dreamshaper-8"
lora = "lora/Chibi Animals.safetensors"
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose",
    torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model, controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
).to("cuda")
torch.cuda.empty_cache()
pipe.enable_xformers_memory_efficient_attention()
pipe.load_lora_weights(lora)


image_path = "input/happy_cat.png"
pose_image_path = "input/control_2.png"

original_image = load_image(image_path)
pose_image = load_image(pose_image_path)

prompt = """
    A unique cute gray cat dancing on the beach. Smiling and having fun.
    Two hands and two legs. A cute little tail. A cute little face.
"""
negative_prompt = "low quality, blurry, distorted, brawny"
height = 896
width = 512


output_image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    init_image=original_image,
    image=pose_image,
    num_inference_steps=50,
    guidance_scale=20,
    strength=0.6,
    height=height,
    width=width
).images[0]

# Save image
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
output_file = "output/" + timestamp + ".png"
output_image.save(output_file)