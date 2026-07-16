#!/usr/bin/env python3
"""
Plot Continuous Distributions
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_continuous_distributions(df, columns_to_plot=None):
    """
    Plot a histogram with KDE and a boxplot for continuous columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to visualize.
    columns_to_plot : list, optional
        Continuous numerical columns to plot. If None, all numerical
        columns are selected.
    """
    if columns_to_plot is None:
        columns_to_plot = df.select_dtypes(
            include=np.number
        ).columns.tolist()

    n_cols = len(columns_to_plot)
    if n_cols == 0:
        return None

    fig, axes = plt.subplots(n_cols, 2, figsize=(10, 3*n_cols))

    if n_cols == 1:
        axes = axes.reshape(1, -1)

    for index, column in enumerate(columns_to_plot):
        # Remove missing values before plotting.
        values = df[column].dropna()

        # Histogram
        axes[index, 0].hist(
            values,
            bins=30,
            density=True,
            alpha=0.7,
            edgecolor="black"
        )

        # KDE curve
        if len(values) > 1 and values.nunique() > 1:
            kde = stats.gaussian_kde(values)

            x_values = np.linspace(
                values.min(),
                values.max(),
                200
            )

            axes[index, 0].plot(
                x_values,
                kde(x_values),
                color="red",
                linestyle="--"
            )

        axes[index, 0].set_title(
            f"{column} Histogram + KDE"
        )

        # Horizontal boxplot
        axes[index, 1].boxplot(
            values,
            vert=False
        )

        axes[index, 1].set_title(
            f"{column} Boxplot"
        )

    plt.tight_layout()
    plt.savefig("Task_8.png")
    plt.show()

    return None
