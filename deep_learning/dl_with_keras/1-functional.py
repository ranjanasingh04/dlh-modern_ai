#!/usr/bin/env python3
"""Build a shallow neural network using the Functional API."""

from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_dim, neurons_h):
    """Return a model with one hidden layer for digit classification."""
    inputs = layers.Input(shape=(input_dim,))
    hidden = layers.Dense(neurons_h, activation="sigmoid")(inputs)
    outputs = layers.Dense(10, activation="softmax")(hidden)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
