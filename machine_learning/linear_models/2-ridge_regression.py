#!/usr/bin/env python3
"""Create a Ridge regression model using Scikit-learn."""

from sklearn import linear_model


def ridge_regression(random_state):
    """Return an untrained Ridge model with the specified random seed."""
    model = linear_model.Ridge(random_state=random_state)
    return model
