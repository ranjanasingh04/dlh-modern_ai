#!/usr/bin/env python3
"""Save and load only the weights of a Keras model."""


def save_model_weights(model, filepath):
    """Saves only the weights of a trained Keras model."""
    model.save_weights(filepath)
    return None


def load_model_weights(model, filepath):
    """Loads saved weights into a compatible Keras model instance."""
    model.load_weights(filepath)
    return None
