#!/usr/bin/env python3
"""Build a shallow neural network for multi-class classification."""

from keras import Sequential, Input
from keras.layers import Dense


def build_model(input_dim, neurons_h):
    """Return a model with one sigmoid hidden layer and softmax output."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(neurons_h, activation="sigmoid"),
        Dense(3, activation="softmax"),
    ])
    return model
