#!/usr/bin/env python3
"""Compute evaluation metrics for a classification model."""
import numpy as np
import sklearn.metrics


def compute_metrics(predictions):
    """Compute accuracy and weighted precision, recall, and F1."""
    logits = predictions.predictions
    labels = predictions.label_ids

    if isinstance(logits, tuple):
        logits = logits[0]

    predicted_labels = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = (
        sklearn.metrics.precision_recall_fscore_support(
            labels,
            predicted_labels,
            average="weighted",
            zero_division=0
        )
    )

    return {
        "accuracy": float(
            sklearn.metrics.accuracy_score(labels, predicted_labels)
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }
