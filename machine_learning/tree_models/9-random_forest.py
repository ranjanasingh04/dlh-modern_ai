#!/usr/bin/env python3
"""Create a random forest classifier using Scikit-learn."""

from sklearn import ensemble


def random_forest(n_estimators, random_state):
    """Return an unfitted random forest with the specified parameters."""
    model = ensemble.RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    return model
