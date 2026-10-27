#!/usr/bin/env python3
"""Generate class predictions from a trained Keras model."""
import tensorflow as tf


def predict(model, X, verbose=0):
    """Makes predictions on a given dataset using a trained Keras model."""
    probabilities = model.predict(X, verbose=verbose)
    predictions = tf.argmax(probabilities, axis=1)
    return predictions
