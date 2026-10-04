#!/usr/bin/env python3
"""Initialize a boosting classifier by algorithm name."""

from sklearn import ensemble
import xgboost as xgb
import lightgbm as lgb


def compare_boosting_classifiers(name, n_estimators, random_state):
    """Return the selected untrained boosting classifier."""
    models = {
        "adaboost": ensemble.AdaBoostClassifier,
        "gradientboosting": ensemble.GradientBoostingClassifier,
        "xgboost": xgb.XGBClassifier,
        "lightgbm": lgb.LGBMClassifier
    }

    if name not in models:
        raise ValueError(f"Unknown model name '{name}'")

    params = {
        "n_estimators": n_estimators,
        "random_state": random_state
    }
    if name == "lightgbm":
        params["verbose"] = -1

    return models[name](**params)
