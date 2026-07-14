#!/usr/bin/env python3
"""
Visualize Missing Data
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """
    Missing plot
    """
    plt.figure(figsize=(12, 8))

    for y, column in enumerate(df.columns):
        missing = (
            df[column].isna() |
            (df[column].astype(str).str.strip() == "")
        )

        plt.scatter(
            np.where(missing)[0],
            np.full(missing.sum(), y),
            marker="|"
        )

    plt.yticks(range(len(df.columns)), df.columns)

    plt.title("Missingness Plot")
    plt.tight_layout()
    plt.show()
