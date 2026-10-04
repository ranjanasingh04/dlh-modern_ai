#!/usr/bin/env python3
"""Create a text-generation pipeline and generate text."""

import transformers


def create_text_generator(model_name, prompt, max_new_tokens,
                          temperature, repetition_penalty,
                          no_repeat_ngram_size):
    """Return the generation pipeline and generated predictions."""
    generator = transformers.pipeline("text-generation", model=model_name)

    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        pad_token_id=generator.tokenizer.eos_token_id,
        do_sample=True
    )

    return generator, output
