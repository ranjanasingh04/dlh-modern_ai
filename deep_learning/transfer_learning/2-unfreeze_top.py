#!/usr/bin/env python3
"""Unfreezes the last layers of a base model."""


def unfreeze_top_layers(model, n_layers):
    """Unfreezes the last n layers of the model's base model.
    """
    base_model = model.layers[0]

    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False

    for layer in base_model.layers[-n_layers:]:
        layer.trainable = True
