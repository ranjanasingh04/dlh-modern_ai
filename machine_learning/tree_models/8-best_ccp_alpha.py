#!/usr/bin/env python3
"""Select the best pruning alpha and its trained decision tree."""


def get_best_alpha(clfs, train_scores, test_scores, ccp_alphas):
    """Select by test accuracy, accuracy gap, then largest alpha."""
    best_index = max(
        range(len(clfs)),
        key=lambda i: (
            test_scores[i],
            -abs(train_scores[i] - test_scores[i]),
            ccp_alphas[i]
        )
    )

    return ccp_alphas[best_index], clfs[best_index]