#!/usr/bin/env python3
"""Evaluate classifier performance using a classification report."""

from sklearn import metrics


def evaluate(true_labels, predicted_labels, class_names):
    """Return a report containing precision, recall, and F1-score."""
    return metrics.classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names
    )
