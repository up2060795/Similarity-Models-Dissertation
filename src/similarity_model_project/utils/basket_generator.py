"""Synthetic basket generator for testing Product2Vec and related models."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from numpy.random import Generator


class BasketGenerator:
    """Generate synthetic shopping baskets for testing and development."""

    def __init__(
        self,
        n_jobs: int | None = None,
        verbose: int = 1,
        seed: int = 1,
        extreme: int | float = 10,
    ) -> None:
        """Initialise the basket generator."""
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.seed = seed
        self.extreme = extreme

        if extreme <= 1:
            raise ValueError("'extreme' must be > 1")

    def _generate_basket(
        self,
        copurchase_probs: np.ndarray,
        min_size: int,
        max_size: int,
        seed: int | None,
    ) -> list[str]:
        """Generate a single synthetic basket using co-purchase probabilities."""
        basket: set[int] = set()
        available_products = np.arange(copurchase_probs.shape[0])

        rng: Generator = np.random.default_rng(seed)
        basket_size = int(rng.integers(min_size, max_size + 1))
        current = int(rng.integers(0, copurchase_probs.shape[0]))
        basket.add(current)

        while len(basket) < basket_size:
            seed = seed + 1 if seed is not None else None
            rng = np.random.default_rng(seed)
            current = int(rng.choice(available_products, p=copurchase_probs[current]))
            basket.add(current)

        return [str(product) for product in sorted(basket)]

    def __call__(
        self,
        n_baskets: int = 1000,
        n_products: int = 100,
        min_size: int = 2,
        max_size: int = 10,
    ) -> list[list[str]]:
        """Generate a full set of synthetic baskets in parallel."""
        if min_size < 2:
            raise ValueError("Minimum basket size is 2.")

        if max_size < min_size:
            raise ValueError("max_size must be greater than or equal to min_size.")

        rng = np.random.default_rng(self.seed)

        matrix = rng.integers(
            low=1,
            high=int(n_products * self.extreme),
            size=(n_products, n_products),
        )
        probs = np.apply_along_axis(lambda x: x / np.sum(x), 1, matrix)

        with Parallel(n_jobs=self.n_jobs, verbose=self.verbose) as parallel:
            return parallel(
                delayed(self._generate_basket)(
                    probs,
                    min_size,
                    max_size,
                    self.seed + i if self.seed is not None else None,
                )
                for i in range(n_baskets)
            )
