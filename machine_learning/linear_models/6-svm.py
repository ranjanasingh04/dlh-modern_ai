#!/usr/bin/env python3
"""Create an SVM classifier with the specified kernel."""

from sklearn import svm


def get_SVM_model(name, random_state):
    """Return an untrained SVC using the requested kernel."""
    model = svm.SVC(
        kernel=name,
        random_state=random_state
    )
    return model
