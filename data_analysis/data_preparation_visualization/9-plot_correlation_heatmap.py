#!/usr/bin/env python3
"""
Plot Correlation Heatmap
"""
import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_heatmap(df):
    """
    Visualizes correlations between continuous numerical features.
    """
    plt.figure(figsize=(6, 5))

    continuous_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    correlation_matrix = df[continuous_columns].corr()

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()
