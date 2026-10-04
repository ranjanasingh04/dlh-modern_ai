#!/usr/bin/env python3
"""Display the decision rules of a trained decision tree."""

from sklearn import tree


def draw(clf, feature_names, class_names):
    """Print the tree structure with feature and class names."""
    rules = tree.export_text(
        clf,
        feature_names=feature_names,
        class_names=class_names
    )
    print(rules)
