#!/usr/bin/env python3
"""Save and load a full Keras model
(architecture, weights, optimizer state)."""
from tensorflow import keras


def save_model(model, filepath):
    """Saves a trained Keras model to the specified filepath."""
    model.save(filepath)
    return None


def load_model(filepath):
    """Loads a full Keras model from the specified filepath."""
    model = keras.models.load_model(filepath)
    return model
