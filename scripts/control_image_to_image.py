import datetime
import random
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, \
    AutoPipelineForImage2Image
from diffusers.utils import load_image


controlnet = ControlNetModel.from_pretrained(
    # "lllyasviel/sd-controlnet-openpose",
    "lllyasviel/control_v11p_sd15_openpose",
    torch_dtype=torch.float16
)

pipe = AutoPipelineForImage2Image.from_pretrained(
    "lykon/dreamshaper-8",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker = None,
    requires_safety_checker = False
).to("cuda")

pipe.load_lora_weights("lora/Chibi Animals.safetensors")
input_image = load_image("input/cat.png")

prompt = "A unique cute gray cat dancing on the beach"
# negative_prompt = "more than 4 legs, more than 2 eyes, more than 1 head, more than 1 tail"

# generator = torch.manual_seed(random.randint(0, 1000))
generator = torch.manual_seed(0)

def generate_image(control_image_path):
    pose_image = load_image(control_image_path)
    image = pipe(
        prompt=prompt,
        # negative_prompt=negative_prompt,
        image=input_image,
        control_image=pose_image,
        num_inference_steps=30,
        # controlnet_conditioning_scale=1.5,
        # guidance_scale=20,
        # strength=0.7,
        generator=generator,
    ).images[0]

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = "output/" + timestamp + ".png"
    image.save(output_file)

generate_image("input/control_1.png")
generate_image("input/control_2.png")