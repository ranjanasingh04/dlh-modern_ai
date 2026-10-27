#!/usr/bin/env python3
"""Train a Keras model."""


def train_model(model, X, Y, epochs, verbose=1):
    """Trains a Keras model."""
    model.fit(X, Y, epochs=epochs, verbose=verbose)
    return None
