#!/usr/bin/env python3
"""Build a shallow neural network for multi-class classification."""

from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_dim, neurons_h):
    """Return a model with one sigmoid hidden layer and softmax output."""
    model = keras.Sequential([
        layers.Dense(neurons_h, activation='sigmoid',
                     input_shape=(input_dim,)),
        layers.Dense(10, activation='softmax')
    ])
    return model
