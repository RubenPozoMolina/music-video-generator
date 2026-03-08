import logging

from utils.modify_utils import ModifyUtils

models = [
    "black-forest-labs/FLUX.2-klein-9B",
    "Qwen/Qwen-Image-Edit"
]

prompts = [
    "The zombie points his guitar towards the moon.",
    "The zombie destroys the guitar"
]

def main():
    logging.basicConfig(level=logging.INFO)
    for model in models:
        modify_utils = ModifyUtils(model)
        from_image = "data/examples/zombie.png"
        for prompt in prompts:
            image = modify_utils.modify_image(
                from_image,
                prompt
            )
            from_image = image
        del modify_utils


if __name__=="__main__":
    main()
