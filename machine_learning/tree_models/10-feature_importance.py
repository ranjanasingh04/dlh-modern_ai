#!/usr/bin/env python3
"""Retrieve and sort random forest feature importances."""

import numpy as np


def feature_importance(rf):
    """Return feature importances and indices in ascending importance."""
    importances = rf.feature_importances_
    indices = np.argsort(importances)
    return importances, indices
