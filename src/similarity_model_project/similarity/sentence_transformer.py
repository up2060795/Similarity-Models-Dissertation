"""Sentence-transformer baseline similarity model."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from huggingface_hub.utils import logging as hf_logging
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as transformers_logging

from similarity_model_project.evaluation.ranking import cosine_similarity
from similarity_model_project.similarity.base import BaseModel, NotFittedError

hf_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()


class SentenceTransformerModel(BaseModel):
    """Pretrained sentence-transformer baseline for product similarity."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialise the sentence-transformer baseline model."""
        super().__init__()
        self.model_name = model_name
        self.encoder: SentenceTransformer | None = None
        self.product_ids_: list[str] = []
        self.embeddings_: dict[str, NDArray[np.float64]] = {}

    def _reset(self) -> None:
        """Clear all learned state so the model can be re-fitted cleanly."""
        super().__init__()
        self.encoder = None
        self.product_ids_ = []
        self.embeddings_ = {}

    def fit(
        self,
        data: Iterator | Iterable[Iterable[str]],
    ) -> SentenceTransformerModel:
        """Build embeddings from (product_id, product_text) pairs."""
        self._reset()

        product_pairs = list(data)
        if len(product_pairs) == 0:
            raise ValueError(
                "No products were provided for sentence-transformer fit()."
            )

        self.product_ids_ = [product_id for product_id, _ in product_pairs]
        texts = [text for _, text in product_pairs]

        self.encoder = SentenceTransformer(
            self.model_name,
            cache_folder=".hf_cache",
        )

        matrix = self.encoder.encode(
            texts,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        matrix = np.asarray(matrix, dtype=np.float64)

        self.embeddings_ = {
            product_id: matrix[i] for i, product_id in enumerate(self.product_ids_)
        }

        self._is_fitted = True
        return self

    def _validate_product_exists(self, product: str) -> None:
        """Check that the product exists in the fitted vocabulary."""
        if not self._is_fitted:
            raise NotFittedError(self)
        if product not in self.embeddings_:
            raise ValueError(
                f"Product '{product}' not found in transformer vocabulary."
            )

    def _topn_from_scores(
        self,
        product: str,
        topn: int,
    ) -> list[tuple[str, float]]:
        """Return the top-N nearest neighbours by cosine similarity."""
        if not self._is_fitted:
            raise NotFittedError(self)

        self._validate_product_exists(product)

        if topn <= 0:
            return []

        focal_vector = self.embeddings_[product]
        scored: list[tuple[str, float]] = []

        for other_product, other_vector in self.embeddings_.items():
            if other_product == product:
                continue
            score = float(cosine_similarity(focal_vector, other_vector))
            scored.append((other_product, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:topn]

    def complements(self, product: str, topn: int = 5) -> list[tuple[str, float]]:
        """Fallback nearest-neighbour complements for baseline compatibility."""
        return self._topn_from_scores(product, topn)

    def substitutes(
        self,
        product: str,
        topn: int = 5,
        penalize: bool = True,
    ) -> list[tuple[str, float]]:
        """Return nearest neighbours as substitute candidates."""
        _ = penalize
        return self._topn_from_scores(product, topn)

    def get_embedding(self, product: str) -> NDArray[np.float64]:
        """Return the embedding vector for a single product."""
        if not self._is_fitted:
            raise NotFittedError(self)

        self._validate_product_exists(product)
        return self.embeddings_[product]

    def get_all_embeddings(self) -> dict[str, NDArray[np.float64]]:
        """Return embeddings for all products in the model vocabulary."""
        if not self._is_fitted:
            raise NotFittedError(self)

        return self.embeddings_
