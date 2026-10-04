#!/usr/bin/env python3
"""Create a Hugging Face fill-mask pipeline."""

import transformers


def fill_mask(model_name, top_k):
    """Return a fill-mask pipeline with the requested top predictions."""
    fill = transformers.pipeline(
        "fill-mask",
        model=model_name,
        tokenizer=model_name,
        top_k=top_k
    )
    return fill
