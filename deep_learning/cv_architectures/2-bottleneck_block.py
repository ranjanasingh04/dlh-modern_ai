#!/usr/bin/env python3
"""Defines a ResNet bottleneck residual block."""

from tensorflow import keras as K


def bottleneck_block(x, filters, stride=1, downsample=False, name=None):
    """
    Builds a ResNet bottleneck residual block.
    """

    def layer_name(suffix):
        """Creates a layer name using the optional block prefix."""
        return f"{name}_{suffix}" if name else None

    shortcut = x

    # 1 × 1 convolution: reduce channels
    y = K.layers.Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        strides=stride,
        padding="valid",
        use_bias=False,
        name=layer_name("conv1")
    )(x)

    y = K.layers.BatchNormalization(
        name=layer_name("bn1")
    )(y)

    y = K.layers.ReLU(
        name=layer_name("relu1")
    )(y)

    # 3 × 3 convolution: process features
    y = K.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        use_bias=False,
        name=layer_name("conv2")
    )(y)

    y = K.layers.BatchNormalization(
        name=layer_name("bn2")
    )(y)

    y = K.layers.ReLU(
        name=layer_name("relu2")
    )(y)

    # 1 × 1 convolution: expand channels by 4
    y = K.layers.Conv2D(
        filters=filters * 4,
        kernel_size=(1, 1),
        strides=1,
        padding="valid",
        use_bias=False,
        name=layer_name("conv3")
    )(y)

    y = K.layers.BatchNormalization(
        name=layer_name("bn3")
    )(y)

    # Projection shortcut when dimensions must be changed
    if downsample:
        shortcut = K.layers.Conv2D(
            filters=filters * 4,
            kernel_size=(1, 1),
            strides=stride,
            padding="valid",
            use_bias=False,
            name=layer_name("downsample_conv")
        )(shortcut)

        shortcut = K.layers.BatchNormalization(
            name=layer_name("downsample_bn")
        )(shortcut)

    # Residual connection
    y = K.layers.Add(
        name=layer_name("add")
    )([y, shortcut])

    # Final activation
    y = K.layers.ReLU(
        name=layer_name("out")
    )(y)

    return y
