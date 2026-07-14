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

    rows, columns = np.where(df.isna())

    plt.scatter(rows, columns, marker="|")
    plt.yticks(
        range(len(df.columns)),
        df.columns
    )
    plt.title("Missingness Plot")

    plt.tight_layout()
    plt.show()
