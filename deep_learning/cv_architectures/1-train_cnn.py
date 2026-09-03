#!/usr/bin/env python3
"""Compiles and trains a convolutional neural network."""

from tensorflow import keras


def compile_and_train_cnn(
    model,
    epochs,
    batch_size,
    x_train,
    y_train,
    x_val,
    y_val,
    optimizer_name='adam',
    optimizer_params=None
):
    """
    Compiles and trains a CNN model.
    """
    if optimizer_params is None:
        optimizer_params = {}

    optimizer = keras.optimizers.get({
        "class_name": optimizer_name,
        "config": optimizer_params
    })

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        verbose=1
    )

    return model, history
