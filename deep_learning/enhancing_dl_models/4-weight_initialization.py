#!/usr/bin/env python3
"""
Weight Initialization

This module implements the build_model_initializer_by_activation function
which builds a Keras model with appropriate weight initializers based on
the activation function used in the hidden layer.

"""

from tensorflow import keras


def build_model_initializer_by_activation(input_dim, hidden_units, activation):
    """
    Build a compiled Keras model with appropriate weight initialization
    based on the activation function.

    This function creates a neural network with a hidden layer using
    an appropriate weight initializer for the specified activation function,
    followed by a softmax output layer.    
    """

    # Step 1: Determine initializer and activation function based on type
    if activation in ["sigmoid", "tanh"]:
        # Glorot Uniform for sigmoid and tanh
        initializer = keras.initializers.GlorotUniform()
        act_func = activation
    elif activation == "relu":
        # He Normal for ReLU
        initializer = keras.initializers.HeNormal()
        act_func = activation
    elif activation == "leaky_relu":
        # He Normal for LeakyReLU
        initializer = keras.initializers.HeNormal()
        # Use LeakyReLU as a layer object instead of string
        act_func = keras.layers.LeakyReLU()
    else:
        raise ValueError(
            "activation must be 'sigmoid', 'tanh', 'relu', or 'leaky_relu'"
        )

    # Step 2: Build model using Functional API
    # Create input layer
    inputs = keras.Input(shape=(input_dim,))

    # Create hidden layer with appropriate initializer and activation
    hidden = keras.layers.Dense(
        units=hidden_units,
        activation=act_func,
        kernel_initializer=initializer,
    )(inputs)

    # Create output layer with softmax activation
    outputs = keras.layers.Dense(units=10, activation="softmax")(hidden)

    # Step 3: Bundle layers into a Model object
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Step 4: Compile the model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
