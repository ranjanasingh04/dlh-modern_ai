#!/usr/bin/env python3
"""
Plots distributions of categorical features.
"""
import matplotlib.pyplot as plt


def plot_categorical_distributions(df, columns_to_plot=None):
    """
    Visualizes the distributions of categorical columns.

    Args:
        df: pandas DataFrame containing categorical columns.
        columns_to_plot: Optional list of categorical columns.
            When None, all object columns except Churn are plotted.

    Returns:
        None
    """
    if columns_to_plot is None:
        columns_to_plot = [
            column for column in df.select_dtypes(include="object").columns
            if column != "Churn"
        ]
    else:
        columns_to_plot = list(columns_to_plot)
    
    n_cols = 3
    n_rows = (len(columns_to_plot) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 5 * n_rows)
    )

    axes = axes.flatten()

    for index, column in enumerate(columns_to_plot):
        df[column].value_counts().plot(
            kind="bar",
            ax=axes[index]
        )

        axes[index].set_title(column)
        axes[index].tick_params(
            axis="x",
            rotation=45
        )

    for index in range(len(columns_to_plot), len(axes)):
        axes[index].axis("off")


    plt.tight_layout()
    plt.savefig("Task_7.png")
    plt.show()
