"""Evaluation metrics for ranked retrieval."""

from __future__ import annotations


def precision_at_k(top_k, relevant, k):
    """Compute precision at k for a ranked retrieval list."""
    if k <= 0:
        return 0.0

    hits = sum(1 for item in top_k[:k] if item in relevant)
    return hits / k
