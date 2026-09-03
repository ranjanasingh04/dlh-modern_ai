#!/usr/bin/env python3
"""Builds the complete MobileNetV1 architecture."""

from tensorflow import keras

mobilenet_backbone = __import__('5-mobilenet_backbone').mobilenet_backbone


def mobilenet(input_shape=(224, 224, 3), num_classes=1000):
    """Builds the MobileNetV1 classification model.
    """
    inputs = keras.Input(shape=input_shape)

    x = mobilenet_backbone(inputs)

    x = keras.layers.GlobalAveragePooling2D()(x)

    outputs = keras.layers.Dense(
        units=num_classes,
        activation='softmax'
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name='MobileNetV1'
    )

    return model
