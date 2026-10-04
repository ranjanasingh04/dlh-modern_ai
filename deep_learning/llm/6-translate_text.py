#!/usr/bin/env python3
"""Create a Hugging Face translation pipeline."""

import transformers


def translate_text(model_name, src_lang="en", tgt_lang="fr"):
    """Return a translation pipeline for the specified languages."""
    translator = transformers.pipeline(
        "translation",
        model=model_name,
        tokenizer=model_name,
        src_lang=src_lang,
        tgt_lang=tgt_lang
    )
    return translator
