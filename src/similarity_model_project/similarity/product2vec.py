"""Product2Vec similarity model.

Product2Vec is based on Word2Vec: instead of learning word embeddings from
sentences, it learns product embeddings from shopping baskets.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator
from functools import partial

import numpy as np
from gensim.models import Word2Vec
from numpy.typing import NDArray
from scipy.optimize import minimize

from similarity_model_project.similarity.base import (
    BaseModel,
    NotFittedError,
    OptimizationWarning,
)


class Product2Vec(BaseModel):
    """Product2Vec model for finding complements and substitutes."""

    def __init__(self, **gensim_kwargs: object) -> None:
        """Initialise the model with Gensim Word2Vec hyperparameters."""
        super().__init__()
        self.gensim_kwargs = gensim_kwargs
        self.model_: Word2Vec | None = None
        self.cond_probs: NDArray[np.float64] | None = None
        self.complementarity_table: NDArray[np.float64] | None = None

    def _reset(self) -> None:
        """Clear all learned state so the model can be re-fitted cleanly."""
        super().__init__()
        self.model_ = None
        self.cond_probs = None
        self.complementarity_table = None

    def _validate_product_in_vocab(self, product: str) -> None:
        """Check that the product exists in the fitted vocabulary."""
        if self.model_ is None:
            raise NotFittedError(self)

        if product not in self.model_.wv.key_to_index:
            raise ValueError(f"Product '{product}' not found in model vocabulary.")

    def _compute_cond_probs(self) -> NDArray[np.float64]:
        """Compute conditional probabilities between all product pairs."""
        if self.model_ is None:
            raise NotFittedError(self)

        def sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
            return 1.0 / (1.0 + np.exp(-values))

        cond_probs = sigmoid(self.model_.syn1neg @ self.model_.wv.vectors.T)
        return np.asarray(cond_probs, dtype=np.float64)

    def _compute_cond_probs_for(self, product_idx: int) -> NDArray[np.float64]:
        """Compute one row of conditional probabilities."""
        if self.model_ is None:
            raise NotFittedError(self)
        row = self.model_.wv.vectors[product_idx]
        return 1.0 / (1.0 + np.exp(-self.model_.syn1neg @ row))

    def _compute_complementarity(
        self,
        product: str,
        immutable: bool = True,
    ) -> NDArray[np.float64]:
        """Get complementarity scores between a focal product and all others."""
        if self.model_ is None:
            raise NotFittedError(self)

        if self.complementarity_table is None:
            self.estimate_complementarity()

        self._validate_product_in_vocab(product)
        assert self.complementarity_table is not None

        product_idx = self.model_.wv.key_to_index[product]
        scores = self.complementarity_table[product_idx, :]

        if immutable:
            copied_scores = scores.copy()
            copied_scores[product_idx] = 0.0
            return copied_scores

        scores[product_idx] = 0.0
        return scores

    def _compute_exchangeability(self, product: str) -> NDArray[np.float64]:
        """Compute how interchangeable a product is with every other product."""
        if self.model_ is None or self.cond_probs is None:
            raise NotFittedError(self)

        self._validate_product_in_vocab(product)
        product_idx = self.model_.wv.key_to_index[product]

        focal_col = self.cond_probs[:, product_idx].reshape(-1, 1)
        probs_diff = focal_col - self.cond_probs
        distances = np.linalg.norm(probs_diff, ord=2, axis=0)
        scores = 1.0 / (1.0 + distances)
        scores[product_idx] = -np.inf

        return np.asarray(scores, dtype=np.float64)

    @staticmethod
    def _abs_corr(
        lambda_param: float | NDArray[np.float64],
        ex_scores: NDArray[np.float64],
        c_scores: NDArray[np.float64],
    ) -> float:
        """Compute absolute Pearson correlation after lambda penalisation."""
        lambda_value = float(np.atleast_1d(lambda_param)[0])
        penalised_scores = ex_scores - lambda_value * c_scores

        penalised_se = np.sqrt(
            np.sum(np.square(penalised_scores - np.mean(penalised_scores)))
        )
        complement_se = np.sqrt(np.sum(np.square(c_scores - np.mean(c_scores))))
        numerator = np.sum(
            (penalised_scores - np.mean(penalised_scores))
            * (c_scores - np.mean(c_scores))
        )
        denominator = penalised_se * complement_se

        if denominator == 0:
            return 0.0

        return float(np.abs(numerator / denominator))

    def _optimize_lambda(
        self,
        c_scores: NDArray[np.float64],
        ex_scores: NDArray[np.float64],
        guess: int | float,
    ) -> float:
        """Find the lambda that best decouples substitutes from complements."""
        mask = np.isfinite(ex_scores) & np.isfinite(c_scores) & (c_scores != 0)
        ex_scores = ex_scores[mask]
        c_scores = c_scores[mask]

        if len(ex_scores) < 2 or len(c_scores) < 2:
            return 0.0

        opt_result = minimize(
            partial(self._abs_corr, ex_scores=ex_scores, c_scores=c_scores),
            x0=float(guess),
        )

        baseline_corr = np.corrcoef(c_scores, ex_scores)[0, 1]
        if np.isnan(baseline_corr):
            baseline_corr = 0.0
        baseline_corr = float(np.abs(baseline_corr))

        if np.abs(float(opt_result.fun)) > baseline_corr:
            warnings.warn(
                OptimizationWarning("Failed to minimize correlation"),
                stacklevel=2,
            )
            return 0.0

        return float(opt_result.x[0])

    def _find_topn(
        self,
        scores: NDArray[np.float64],
        topn: int,
    ) -> list[tuple[str, float]]:
        """Return the top-N products by score."""
        if self.model_ is None:
            raise NotFittedError(self)

        if topn <= 0:
            return []

        valid_mask = np.isfinite(scores)
        valid_indices = np.where(valid_mask)[0]
        valid_scores = scores[valid_mask]

        if len(valid_scores) == 0:
            return []

        best_local = np.argsort(valid_scores)[::-1][:topn]
        best_candidates = valid_indices[best_local]

        labels = np.array(self.model_.wv.index_to_key)
        candidates = list(zip(labels[best_candidates], scores[best_candidates]))
        return [(str(label), float(score)) for label, score in candidates]

    def fit(
        self, data: Iterator[Iterable[str]] | Iterable[Iterable[str]]
    ) -> Product2Vec:
        """Train Word2Vec on shopping baskets to learn product embeddings."""
        self._reset()

        baskets = data
        self.model_ = Word2Vec(
            sentences=baskets,
            sg=1,
            hs=0,
            shrink_windows=False,
            **self.gensim_kwargs,
        )

        self._is_fitted = True
        return self

    def estimate_complementarity(self) -> Product2Vec:
        """Compute and store the complementarity table after fitting."""
        if self.model_ is None:
            raise NotFittedError(self)

        self.cond_probs = self._compute_cond_probs()
        self.complementarity_table = 0.5 * (self.cond_probs + self.cond_probs.T)
        return self

    def complements(self, product: str, topn: int = 5) -> list[tuple[str, float]]:
        """Return the top-N complementary products."""
        if not self._is_fitted:
            raise NotFittedError(self)

        scores = self._compute_complementarity(product)
        return self._find_topn(scores, topn)

    def substitutes(
        self,
        product: str,
        topn: int = 10,
        penalize: bool = True,
        guess: int | float = 0,
    ) -> list[tuple[str, float]]:
        """Return the top-N substitute products."""
        if not self._is_fitted:
            raise NotFittedError(self)

        if self.cond_probs is None:
            self.estimate_complementarity()

        ex_scores = self._compute_exchangeability(product)

        if penalize:
            c_scores = self._compute_complementarity(product)
            lambda_param = self._optimize_lambda(
                c_scores=c_scores,
                ex_scores=ex_scores,
                guess=guess,
            )
            ex_scores = ex_scores - lambda_param * c_scores

        return self._find_topn(ex_scores, topn)

    def get_embedding(self, product: str) -> NDArray[np.float64]:
        """Return the embedding vector for a single product."""
        if not self._is_fitted or self.model_ is None:
            raise NotFittedError(self)

        self._validate_product_in_vocab(product)
        return np.asarray(self.model_.wv[product], dtype=np.float64)

    def get_all_embeddings(self) -> dict[str, NDArray[np.float64]]:
        """Return embeddings for all products in the model vocabulary."""
        if not self._is_fitted or self.model_ is None:
            raise NotFittedError(self)

        return {
            product: np.asarray(self.model_.wv[product], dtype=np.float64)
            for product in self.model_.wv.index_to_key
        }
