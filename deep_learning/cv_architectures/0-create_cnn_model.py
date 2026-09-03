#!/usr/bin/env python3
"""Create a convolutional neural network."""

from tensorflow import keras


def create_cnn_model(input_shape, filters, kernel_sizes,
                     activations, pooling_type='max'):
    """
    Creates and compiles a CNN classification model.
    """
    if not (len(filters) == len(kernel_sizes) == len(activations)):
        raise ValueError(
            "filters, kernel_sizes and activations must have equal lengths"
        )

    if pooling_type not in ('max', 'avg'):
        raise ValueError("pooling_type must be 'max' or 'avg'")

    model = keras.Sequential()

    for index, (number_of_filters, kernel_size, activation) in enumerate(
        zip(filters, kernel_sizes, activations)
    ):
        if index == 0:
            model.add(
                keras.layers.Conv2D(
                    filters=number_of_filters,
                    kernel_size=kernel_size,
                    activation=activation,
                    input_shape=input_shape
                )
            )
        else:
            model.add(
                keras.layers.Conv2D(
                    filters=number_of_filters,
                    kernel_size=kernel_size,
                    activation=activation
                )
            )

        if pooling_type == 'max':
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        else:
            model.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
