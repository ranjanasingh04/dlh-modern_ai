#!/usr/bin/env python3
"""Creates an image data augmentation pipeline."""

import tensorflow as tf


def build_data_augmentation():
    """Builds a reproducible image augmentation model.

    Returns:
        A Keras Sequential model containing augmentation layers.
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(
            mode='horizontal',
            seed=42
        ),
        tf.keras.layers.RandomRotation(
            factor=0.15,
            seed=42
        ),
        tf.keras.layers.RandomZoom(
            height_factor=0.15,
            seed=42
        ),
        tf.keras.layers.RandomContrast(
            factor=0.1,
            seed=42
        )
    ])

    return data_augmentation
