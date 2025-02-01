import datetime
import torch
from diffusers import StableDiffusionPipeline

model = "lykon/dreamshaper-8"
lora = "lora/Chibi Animals.safetensors"
prompt = """
    A unique cute gray cat dancing on the beach. Smiling and having fun.
    Two hands and two legs. A cute little tail. A cute little face.
"""
height = 896
width = 512

# Prepare the pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    model, torch_dtype=torch.float16
).to("cuda")
pipe.load_lora_weights(lora)
torch.cuda.empty_cache()
pipe.enable_xformers_memory_efficient_attention()

# Generate image
image = pipe(
    prompt=prompt,
    num_inference_steps=50,
    height=height,
    width=width,
    strength=0.5
).images[0]

# Save image
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
output_file = "output/" + timestamp + ".png"
image.save(output_file)