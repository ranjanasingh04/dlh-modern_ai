#!/usr/bin/env python3
"""
Task 5: L2 Regularization

This module implements the build_model_with_L2_regularization function
which builds a Keras model with L2 regularization applied to the kernel
weights of all hidden layers.

L2 regularization penalizes large weights to prevent overfitting and
encourage the model to learn simpler, more generalizable patterns.
"""

from tensorflow import keras


def build_model_with_L2_regularization(input_dim, hidden_units,
                                       n_layers, lambda_l2):
    """
    Build a Keras model with L2 regularization on hidden layers.

    This function creates a neural network with multiple hidden layers,
    each using ReLU activation and L2 regularization on kernel weights,
    followed by a softmax output layer.
    """

    # Input validation
    if n_layers < 1:
        raise ValueError("n_layers must be at least 1")
    if lambda_l2 < 0:
        raise ValueError("lambda_l2 must be non-negative")

    # Create L2 regularizer (always, even if lambda_l2 is 0)
    l2_regularizer = keras.regularizers.L2(lambda_l2)

    # Step 1: Build model using Functional API
    # Create input layer
    inputs = keras.Input(shape=(input_dim,))

    # Step 2: Add hidden layers with L2 regularization
    x = inputs
    for i in range(n_layers):
        x = keras.layers.Dense(
            units=hidden_units,
            activation='relu',
            kernel_regularizer=l2_regularizer,
            name=f'hidden_layer_{i+1}'
        )(x)

    # Step 3: Add output layer (no regularization on output)
    outputs = keras.layers.Dense(
        units=10,
        activation='softmax',
        name='output_layer'
    )(x)

    # Step 4: Bundle layers into a Model object
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Step 5: Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model
