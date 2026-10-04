#!/usr/bin/env python3
"""Generate image captions using a pretrained BLIP model."""

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


def image_captioner(model, image_path, max_new_tokens):
    """Return a caption for the image using the specified BLIP model."""
    processor = BlipProcessor.from_pretrained(model)
    caption_model = BlipForConditionalGeneration.from_pretrained(model)
    caption_model.eval()

    with Image.open(image_path) as image:
        inputs = processor(
            images=image.convert("RGB"),
            return_tensors="pt"
        )

    with torch.no_grad():
        output = caption_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption
