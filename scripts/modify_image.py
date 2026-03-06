import logging

from utils.modify_utils import ModifyUtils


def main():
    modify_utils = ModifyUtils()
    zombie = modify_utils.modify_image(
        "data/examples/zombie.png",
        "The zombie raise the guitar to the moon"
    )
    logging.info(f"Zombie saved to {zombie}")

if __name__=="__main__":
    main()
