#!/usr/bin/env python3
"""Compute raw logits for masked tokens using RoBERTa."""

import torch


def compute_mask_logits(model, inputs, mask_indices):
    """Return a logits tensor for each mask position."""
    with torch.no_grad():
        outputs = model(**inputs)
        mask_logits_list = [
            outputs.logits[0, index, :] for index in mask_indices
        ]

    return mask_logits_list
