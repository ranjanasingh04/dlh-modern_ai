#!/usr/bin/env python3
"""Adds a classification head to a feature extraction model."""

from tensorflow import keras


def add_classification_head(base_model, num_classes):
    """Adds a custom classification head to a base model.
    """
    x = base_model.output

    x = keras.layers.Dense(
        units=128,
        activation='relu'
    )(x)

    outputs = keras.layers.Dense(
        units=num_classes,
        activation='softmax'
    )(x)

    model = keras.Model(
        inputs=base_model.input,
        outputs=outputs
    )

    return model
