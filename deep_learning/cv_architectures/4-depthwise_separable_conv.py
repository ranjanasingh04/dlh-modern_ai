#!/usr/bin/env python3
"""Implements a depthwise separable convolution block."""

from tensorflow import keras


def depthwise_separable_conv(X, filters, stride=1):
    """Builds a MobileNetV1 depthwise separable convolution block.
    """
    # Apply one 3x3 spatial filter to each input channel separately
    x = keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding='same',
        use_bias=False
    )(X)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    # Combine channel information using a 1x1 convolution
    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=False
    )(x)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    return x
