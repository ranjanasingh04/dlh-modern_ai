#!/usr/bin/env python3
"""Builds the ResNet-101 architecture."""

from tensorflow import keras as K

bottleneck_block = __import__(
    '2-bottleneck_block'
).bottleneck_block


def make_layer(x, blocks, filters, stride=1, name=None):
    """
    Builds one stage of ResNet bottleneck blocks.

    Args:
        x: Input tensor.
        blocks: Number of bottleneck blocks in the stage.
        filters: Number of filters in the block's 3x3 convolution.
        stride: Stride used by the first block.
        name: Name prefix for the stage.

    Returns:
        Output tensor of the completed stage.
    """
    x = bottleneck_block(
        x,
        filters=filters,
        stride=stride,
        downsample=True,
        name=f"{name}_block1"
    )

    for i in range(1, blocks):
        x = bottleneck_block(
            x,
            filters=filters,
            stride=1,
            downsample=False,
            name=f"{name}_block{i + 1}"
        )

    return x


def build_resnet101(input_shape=(224, 224, 3), num_classes=1000):
    """
    Builds a ResNet-101 model.
    """
    inputs = K.Input(shape=input_shape)

    # Initial convolution
    x = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=2,
        padding="same",
        use_bias=False,
        name="conv1"
    )(inputs)

    x = K.layers.BatchNormalization(
        name="bn1"
    )(x)

    x = K.layers.ReLU(
        name="relu1"
    )(x)

    # Initial max pooling
    x = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=2,
        padding="same",
        name="maxpool"
    )(x)

    # conv2_x: 3 bottleneck blocks
    x = make_layer(
        x,
        blocks=3,
        filters=64,
        stride=1,
        name="layer1"
    )

    # conv3_x: 4 bottleneck blocks
    x = make_layer(
        x,
        blocks=4,
        filters=128,
        stride=2,
        name="layer2"
    )

    # conv4_x: 23 bottleneck blocks
    x = make_layer(
        x,
        blocks=23,
        filters=256,
        stride=2,
        name="layer3"
    )

    # conv5_x: 3 bottleneck blocks
    x = make_layer(
        x,
        blocks=3,
        filters=512,
        stride=2,
        name="layer4"
    )

    # Classification head
    x = K.layers.GlobalAveragePooling2D(
        name="avgpool"
    )(x)

    outputs = K.layers.Dense(
        units=num_classes,
        activation="softmax",
        name="fc"
    )(x)

    model = K.Model(
        inputs=inputs,
        outputs=outputs,
        name="resnet101"
    )

    return model
