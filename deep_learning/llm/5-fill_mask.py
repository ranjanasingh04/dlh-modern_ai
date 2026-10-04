#!/usr/bin/env python3
"""Create a Hugging Face fill-mask pipeline."""

from transformers import pipeline


def fill_mask(model_name, top_k):
    """Return a fill-mask pipeline with the requested top predictions."""
    fill = pipeline(
        "fill-mask",
        model=model_name,
        tokenizer=model_name,
        top_k=top_k
    )
    return fill
