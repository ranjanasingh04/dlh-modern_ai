#!/usr/bin/env python3
"""Builds a frozen MobileNetV2 feature extractor."""

from tensorflow import keras


def build_feature_extractor():
    """Builds a frozen MobileNetV2 feature extraction model.
    """
    base_model = keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))

    x = base_model(inputs, training=False)

    outputs = keras.layers.GlobalAveragePooling2D()(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    return model
