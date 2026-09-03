#!/usr/bin/env python3
"""Performs agglomerative hierarchical clustering."""

from sklearn import cluster
from sklearn import metrics

Apply_PCA = __import__('1-pca').Apply_PCA


def Agglomerative_Clustering(X, n_clusters, random_state, n_components,
                             use_pca_data=True):
    """Performs agglomerative clustering and evaluates the result.
    """
    if use_pca_data:
        X_used, _ = Apply_PCA(
            X,
            n_components=n_components,
            random_state=random_state
        )
    else:
        X_used = X

    model = cluster.AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )

    model.fit(X_used)

    if n_clusters > 1:
        score = metrics.silhouette_score(
            X_used,
            model.labels_
        )
    else:
        score = None

    return model, X_used, score
