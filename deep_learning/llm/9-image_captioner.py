#!/usr/bin/env python3
"""Generate image captions using a pretrained BLIP model."""

import torch
import PIL
import transformers


def image_captioner(model, image_path, max_new_tokens):
    """Return a caption for the image using the specified BLIP model."""
    processor = transformers.BlipProcessor.from_pretrained(model)
    caption_model = transformers.BlipForConditionalGeneration.from_pretrained(
        model)
    caption_model.eval()

    with PIL.Image.open(image_path) as image:
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
