#!/usr/bin/env python3
"""Find mask token positions in a tokenized input sequence."""


def get_mask_index(inputs, tokenizer):
    """Return all mask indices, raising ValueError if none exist."""
    token_ids = inputs["input_ids"][0]
    mask_indices = (
        (token_ids == tokenizer.mask_token_id)
        .nonzero(as_tuple=True)[0]
        .tolist()
    )

    if not mask_indices:
        raise ValueError("No <mask> token found in the input!")

    return mask_indices
