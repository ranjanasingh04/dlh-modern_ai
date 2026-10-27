#!/usr/bin/env python3
"""Momentum-Based SGD Variants Optimization"""

from tensorflow import keras


def get_optimizer_SGD(name, lr, momentum=0.0, nesterov=False):
    """
    Configure and return an SGD-based optimizer with momentum variants.

    This function creates an SGD optimizer with the specified learning rate
    and momentum settings based on the selected variant.
    """

    # Configure SGD optimizer based on selected variant
    if name == 'SGD':
        # Standard SGD: no momentum, no Nesterov
        optimizer = keras.optimizers.SGD(
            learning_rate=lr,
            momentum=0.0,
            nesterov=False
        )
    elif name == 'SGD+Momentum':
        # SGD with classical momentum
        optimizer = keras.optimizers.SGD(
            learning_rate=lr,
            momentum=momentum,
            nesterov=False
        )
    elif name == 'SGD+Momentum+Nesterov':
        # SGD with momentum and Nesterov acceleration
        optimizer = keras.optimizers.SGD(
            learning_rate=lr,
            momentum=momentum,
            nesterov=True
        )
    else:
        # Raise error for invalid variant
        raise ValueError(
            "name must be 'SGD', 'SGD+Momentum', or 'SGD+Momentum+Nesterov'"
        )

    return optimizer
