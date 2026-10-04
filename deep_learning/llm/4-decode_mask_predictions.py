#!/usr/bin/env python3
"""Decode vocabulary tokens for every masked position."""


def decode_mask_predictions(mask_logits_list, tokenizer):
    """Return vocabulary strings in token ID order for each mask."""
    decoded_tokens = [
        [
            tokenizer.decode([token_id]).strip()
            for token_id in range(mask_logits.shape[-1])
        ]
        for mask_logits in mask_logits_list
    ]
    return decoded_tokens
