"""Ranking helpers for embedding-based evaluation."""

from __future__ import annotations

import numpy as np


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def rank_by_cosine_similarity(
    focal_item_id: str,
    embeddings: dict[str, np.ndarray],
    topn: int,
) -> list[tuple[str, float]]:
    """Rank all other items by cosine similarity to the focal item."""
    if focal_item_id not in embeddings:
        raise KeyError(f"Focal item '{focal_item_id}' not found in embeddings.")

    item_ids = list(embeddings.keys())
    matrix = np.array([embeddings[i] for i in item_ids], dtype=np.float32)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms

    focal_idx = item_ids.index(focal_item_id)
    scores = matrix @ matrix[focal_idx]
    scores[focal_idx] = -np.inf

    topn = min(topn, len(item_ids) - 1)
    top_indices = np.argpartition(scores, -topn)[-topn:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return [(item_ids[i], float(scores[i])) for i in top_indices]
