from utils.image_generator import ImageGenerator


class TestImageGenerator:

    def test_generate_image(self):
        image_generator = ImageGenerator("zombie playing guitar in a dark night")
        image = image_generator.generate("output")
        assert image.size == (512, 512)
