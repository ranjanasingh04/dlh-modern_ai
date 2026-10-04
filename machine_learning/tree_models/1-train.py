#!/usr/bin/env python3
"""Train a Scikit-learn tree-based classifier."""


def train_tree(clf, X, y):
    """Fit the classifier using features X and target labels y."""
    clf.fit(X, y)
