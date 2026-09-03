#!/usr/bin/env python3
"""Evaluates different numbers of K-Means clusters."""

from sklearn import metrics

K_Means = __import__('2-k_means').K_Means


def optimal_k(X, max_clusters, random_state):
    """Evaluates K-Means models using inertia and silhouette score.
    """
    ks = list(range(2, max_clusters + 1))
    inertia_values = []
    silhouette_values = []

    for k in ks:
        model = K_Means(
            X,
            n_clusters=k,
            random_state=random_state
        )

        labels = model.labels_

        inertia_values.append(model.inertia_)

        score = metrics.silhouette_score(X, labels)
        silhouette_values.append(score)

    return ks, inertia_values, silhouette_values
