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

    corr = df.corr(numeric_only=True)
    sns.heatmap(corr,
                vmin=-1,
                vmax=1,
                annot=True,
                cmap='coolwarm'
                )
    plt.title('Correlation Matrix')
    plt.show()
