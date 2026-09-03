#!/usr/bin/env python3
"""Standardizes tabular features using Scikit-learn."""

from sklearn import preprocessing


def Standardize(X):
    """Standardizes every feature in a tabular dataset.
    """
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled
