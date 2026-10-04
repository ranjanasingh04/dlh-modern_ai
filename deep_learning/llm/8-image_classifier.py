#!/usr/bin/env python3
"""Create a Hugging Face image-classification pipeline."""

import transformers


def image_classifier(model):
    """Return an image classifier using the specified pretrained model."""
    classifier = transformers.pipeline("image-classification", model=model)
    return classifier
