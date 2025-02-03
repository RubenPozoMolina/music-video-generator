import datetime

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, \
    UniPCMultistepScheduler
from diffusers.utils import load_image

from utils.file_utils import get_all_files

model_id = "lykon/dreamshaper-8"
lora = "lora/Chibi Animals.safetensors"
files = get_all_files("input/controls")
init_image = "input/cat.png"
prompt = """A unique cute gray cat dancing on the beach. 
Smiling and having fun. 
Two hands and two legs. 
A cute little tail. A cute little face. flat horizon. Noon.
Same background as input image.
Same character as input image.
"""
negative_prompt = "low quality, blurry, distorted, brawny, malformations, trousers, gloves"
images_path = "output/images"


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
# generator = torch.manual_seed(0)
for file in files:
    control_image = load_image(file)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        init_image=source_image,
        image=control_image,
        num_inference_steps=50,
        guidance_scale=7,
        strength=0.7,
        # generator=generator
    ).images[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = "output/" + timestamp + ".png"
    image.save(output_file)
    # source_image = image
