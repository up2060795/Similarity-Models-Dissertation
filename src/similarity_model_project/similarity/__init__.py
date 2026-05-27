"""Similarity models package."""

from similarity_model_project.similarity.base import (
    BaseModel,
    NotFittedError,
    OptimizationWarning,
)
from similarity_model_project.similarity.product2vec import Product2Vec

__all__ = [
    "BaseModel",
    "NotFittedError",
    "OptimizationWarning",
    "Product2Vec",
]
