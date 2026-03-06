from utils.image_utils import ImageUtils


def main():
    image_utils = ImageUtils()
    zombie = image_utils.create_image(
        "Lykon/DreamShaper",
        "A zombie playing an electric guitar in a dark cemetery under the moonlight",
        512,
        512

    )

if __name__=="__main__":
    main()
