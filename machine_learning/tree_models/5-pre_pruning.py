#!/usr/bin/env python3
"""Find the best decision tree pre-pruning hyperparameters."""

from sklearn import model_selection


def prepruning(X, y, clf):
    """Return the best hyperparameters found through grid search."""
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": range(2, 5),
        "min_samples_leaf": range(2, 5),
        "min_samples_split": range(2, 5)
    }

    search = model_selection.GridSearchCV(clf, param_grid)
    search.fit(X, y)

    return search.best_params_
