#!/usr/bin/env python3
"""Create a Lasso regression model using Scikit-learn."""

from sklearn import linear_model


def lasso_regression(random_state):
    """Return an untrained Lasso model with the specified random seed."""
    model = linear_model.Lasso(random_state=random_state)
    return model
