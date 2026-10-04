#!/usr/bin/env python3
"""Tokenize text using a pretrained RoBERTa tokenizer."""

import transformers


def tokenize_text(model_name, sentence, padding=True):
    """Return the RoBERTa tokenizer and tokenized PyTorch inputs."""
    tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)
    inputs = tokenizer(
        sentence,
        padding=padding,
        return_tensors="pt"
    )
    return tokenizer, inputs
