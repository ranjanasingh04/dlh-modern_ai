#!/usr/bin/env python3
"""Visualize class counts and raw SMS message lengths."""

import matplotlib.pyplot as plt
import seaborn as sns


def explore_data(df):
    """Plot ham/spam counts and a histogram of raw message lengths."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    counts = df['label'].value_counts().reindex(['ham', 'spam'], fill_value=0)
    sns.barplot(x=counts.index, y=counts.values, ax=ax1)
    ax1.set(title='Ham vs Spam Counts', xlabel='label', ylabel='count')

    lengths = df['message'].str.len()
    sns.histplot(lengths, bins=50, ax=ax2)
    ax2.set(title='Histogram of Raw Message Lengths',
            xlabel='length', ylabel='count')

    plt.tight_layout()
