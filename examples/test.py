"""for Win10
pip install bitsandbytes-windows
useful link: https://huggingface.co/CompVis/stable-diffusion-v1-4?text=a+photo+of+blone+and+cute+girl
"""
from unittest import TestCase
from aphrodite.action import Draw


class DrawTest(TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_draw(self) -> None:
        draw = Draw()
        prompt = "a photo of an astronaut riding a horse on mars"

        image = draw.do_action(prompt=prompt, img_shape=img_shape)
        self.assertEqual(type(image), "PIL.Image.Image")


if __name__ == "__main__":
    draw = Draw()

    img_shape = (256, 256)
    prompt = "a photo of an astronaut riding a horse on mars"
    image = draw.do_action(prompt=prompt)  # , img_shape=img_shape)
    image.save("./test.png", "png")
