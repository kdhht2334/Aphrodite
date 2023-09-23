import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

import torch
from diffusers import StableDiffusionPipeline

from aphrodite.action import Action


class Draw(Action):
    def __init__(self) -> None:
        super().__init__()

        self._model_id = "CompVis/stable-diffusion-v1-4"
        self._device = "cuda"

    def do_action(self, prompt: str, img_shape: tuple = None) -> None:
        pipe = StableDiffusionPipeline.from_pretrained(
            self._model_id,
            # safety_checker=None,
            # requires_safety_checker=False,
            revision="fp16",
            torch_dtype=torch.float16,
        )
        pipe = pipe.to(self._device)
        pipe.enable_attention_slicing()

        if img_shape is not None:
            image = pipe(prompt, height=img_shape[0], width=img_shape[1]).images[0]
        else:
            image = pipe(prompt).images[0]
        return image
