#!/usr/bin/env python3
"""Create a decision tree classifier using Scikit-learn."""

from sklearn import tree


def build_decision_tree(min_samples_leaf, min_samples_split, random_state):
    """Return an unfitted decision tree classifier using Gini impurity."""
    model = tree.DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    return model
