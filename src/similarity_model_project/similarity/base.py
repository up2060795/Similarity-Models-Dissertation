"""Base abstractions and custom exceptions for similarity models.

This module defines the contract that every similarity model must follow.
Any new model (e.g. a transformer-based one) must inherit from BaseModel
and implement the required abstract methods. This guarantees that the rest
of the pipeline can swap models in and out without changing surrounding code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

import numpy as np


class BaseModel(ABC):
    """Abstract base class that every similarity model must inherit from."""

    def __init__(self) -> None:
        """Initialise the fitted flag to False."""
        self._is_fitted: bool = False

    @abstractmethod
    def _reset(self) -> None:
        """Wipe all learned attributes before re-fitting."""

    @abstractmethod
    def fit(
        self,
        data: (
            Iterator[Iterable[str]]  # Product2Vec (baskets)
            | Iterable[Iterable[str]]
            | Iterator[tuple[str, str]]  # Sentence Transformer (id, text)
            | Iterable[tuple[str, str]]
        ),
    ) -> BaseModel:
        """Fit the model using model-specific input data."""

    @abstractmethod
    def complements(self, product: str, topn: int = 5) -> list[tuple[str, float]]:
        """Find the top-N complementary products."""

    @abstractmethod
    def substitutes(
        self,
        product: str,
        topn: int = 5,
        penalize: bool = True,
    ) -> list[tuple[str, float]]:
        """Find the top-N substitute products."""

    @abstractmethod
    def get_embedding(self, product: str) -> np.ndarray:
        """Return the embedding vector for a single product."""

    @abstractmethod
    def get_all_embeddings(self) -> dict[str, np.ndarray]:
        """Return embeddings for all products in the model vocabulary."""


class NotFittedError(Exception):
    """Raised when a model method is called before fit()."""

    def __init__(self, cls: object) -> None:
        """Initialize the exception with the class name that is not fitted."""
        msg = f"{cls.__class__.__name__} is not fitted"
        super().__init__(msg)


class OptimizationWarning(UserWarning):
    """Issued when lambda optimisation fails to improve results."""

    def __init__(self, msg: str) -> None:
        """Initialize the warning with a custom message."""
        super().__init__()
        self.msg = msg

    def __str__(self) -> str:
        """Return the warning message as a string."""
        return self.msg
