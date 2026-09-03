#!/usr/bin/env python3
"""Unfreezes the last layers of a base model."""


def unfreeze_top_layers(model, n_layers):
    """Unfreezes the last n layers of the model's base model.
    """
    if n_layers <= 0:
        raise ValueError("n must be a positive integer")
    if n_layers > len(model.layers):
        raise ValueError("n must be less than or equal "
                         "to the number of layers in the base model")

    # Unfreeze the last N layers
    for layer in model.layers[-n_layers:]:
        layer.trainable = True
