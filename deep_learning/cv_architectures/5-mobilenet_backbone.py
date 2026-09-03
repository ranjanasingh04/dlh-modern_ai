#!/usr/bin/env python3
"""Builds the MobileNetV1 feature extraction backbone."""

from tensorflow import keras

depthwise_separable_conv = __import__(
    '4-depthwise_separable_conv'
).depthwise_separable_conv


def mobilenet_backbone(inputs):
    """Builds the feature extraction backbone of MobileNetV1.

    Args:
        inputs: Input tensor to the network.

    Returns:
        Output tensor of the MobileNetV1 backbone.
    """
    # Initial standard 3x3 convolution
    x = keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        strides=2,
        padding='same',
        use_bias=False
    )(inputs)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    # MobileNetV1 depthwise separable convolution blocks
    x = depthwise_separable_conv(x, filters=64, stride=1)

    x = depthwise_separable_conv(x, filters=128, stride=2)
    x = depthwise_separable_conv(x, filters=128, stride=1)

    x = depthwise_separable_conv(x, filters=256, stride=2)
    x = depthwise_separable_conv(x, filters=256, stride=1)

    x = depthwise_separable_conv(x, filters=512, stride=2)

    for _ in range(5):
        x = depthwise_separable_conv(x, filters=512, stride=1)

    x = depthwise_separable_conv(x, filters=1024, stride=2)
    x = depthwise_separable_conv(x, filters=1024, stride=1)

    return x
