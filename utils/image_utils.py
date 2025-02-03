import torch
from diffusers import ControlNetModel, AutoPipelineForImage2Image
from diffusers.utils import load_image


def generate_image(
        init_image_path, control_image_path,
        model_id, lora, prompt, negative_prompt
):
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose",
        torch_dtype=torch.float16
    )

    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to("cuda")

    pipe.load_lora_weights(lora)
    init_image = load_image(init_image_path)

    control_image = load_image(control_image_path)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        init_image=init_image,
        image=control_image,
        num_inference_steps=50,
        guidance_scale=7,
        strength=0.7,
    ).images[0]
    return image
