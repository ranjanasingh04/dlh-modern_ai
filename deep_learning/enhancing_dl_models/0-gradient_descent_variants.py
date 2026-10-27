#!/usr/bin/env python3
"""Gradient Descent Variants Optimization"""

from tensorflow import keras


def train_with_gradient_descent_variant(variant, learning_rate,
                                        x_train, batch_size):
    """
    Configure and return a gradient descent optimizer based on the variant.

    This function creates an SGD optimizer with the specified learning rate
    and determines the appropriate batch size based on the selected gradient
    descent variant strategy.
    """

    # Create SGD optimizer with specified learning rate
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

    # Determine batch size based on selected variant
    if variant == 'batch':
        # Use entire dataset as one batch
        bs = len(x_train)
    elif variant == 'stochastic':
        # Use single sample per update
        bs = 1
    elif variant == 'mini_batch':
        # Use custom batch size provided by user
        bs = batch_size
    else:
        # Raise error for invalid variant
        raise ValueError(
            "variant must be 'batch', 'stochastic', or 'mini_batch'"
        )

    return optimizer, bs
