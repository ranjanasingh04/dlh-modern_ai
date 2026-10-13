#!/usr/bin/env python3
"""Tokenize the Emotion dataset with a pretrained tokenizer."""


def tokenize_and_map(dataset, tokenizer, max_length, truncation, batched):
    """Tokenize the text column across all dataset splits."""

    def tokenize_examples(examples):
        return tokenizer(
            examples["text"],
            max_length=max_length,
            truncation=truncation
        )

    tokenized = dataset.map(tokenize_examples, batched=batched)
    return tokenized["train"], tokenized["validation"], tokenized["test"]
