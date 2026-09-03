#!/usr/bin/env python3
"""Unfreezes the last layers of a pretrained model."""


def unfreeze_top_layers(model, n_layers):
    """Unfreezes the last n layers of a model.

    Args:
        model: Keras model whose layers will be modified.
        n_layers: Number of layers to unfreeze.

    Returns:
        None.
    """
    model.trainable = True

    total_layers = len(model.layers)
    n_layers = max(0, min(n_layers, total_layers))
    cutoff = total_layers - n_layers

    for index, layer in enumerate(model.layers):
        layer.trainable = index >= cutoff
