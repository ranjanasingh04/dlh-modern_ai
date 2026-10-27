#!/usr/bin/env python3
"""Compile a Keras model with SGD, binary cross-entropy, and accuracy."""
from tensorflow import keras


def compile_model(model, learning_rate=0.01):
    """
    Configures a Keras model for training."""
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return None
