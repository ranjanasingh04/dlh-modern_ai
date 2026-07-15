#!/usr/bin/env python3
"""
Plot Target Distribution
"""
import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """
    Plot Target Distribution
    """
    plt.figure(figsize=(12, 8))

    counts = df["Churn"].value_counts()
    plt.bar(
        counts.index,
        counts.values,
        color=["skyblue", "salmon"]
    )
    plt.title("Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Count")

    plt.show
