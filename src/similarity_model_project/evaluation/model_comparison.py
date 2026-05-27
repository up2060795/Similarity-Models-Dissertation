from __future__ import annotations

from typing import Any

import polars as pl

from similarity_model_project.evaluation.ranking import rank_by_cosine_similarity
from similarity_model_project.similarity.base import BaseModel


def get_product_metadata(
    mapping_df: pl.DataFrame,
    item_id: str,
) -> dict[str, str]:
    """Return product name and category for an item ID."""
    match = mapping_df.filter(pl.col("ITEM_ID").cast(pl.String) == str(item_id))

    if match.height == 0:
        return {
            "ITEM_ID": str(item_id),
            "PRODUCT_NAME": str(item_id),
            "CATEGORY": "UNKNOWN_CATEGORY",
        }

    row = match.row(0, named=True)

    return {
        "ITEM_ID": str(item_id),
        "PRODUCT_NAME": str(row.get("PRODUCT_NAME", item_id)),
        "CATEGORY": str(row.get("CATEGORY", "UNKNOWN_CATEGORY")),
    }


def get_top_k_by_cosine(
    model: BaseModel,
    anchor_item_id: str,
    mapping_df: pl.DataFrame,
    k: int,
) -> list[dict[str, Any]]:
    """Return top-k cosine neighbours for one model and one anchor product."""
    embeddings = model.get_all_embeddings()

    if anchor_item_id not in embeddings:
        raise ValueError(
            f"Anchor item '{anchor_item_id}' not found in model embeddings."
        )

    ranked_items = rank_by_cosine_similarity(
        focal_item_id=anchor_item_id,
        embeddings=embeddings,
        topn=k,
    )

    rows: list[dict[str, Any]] = []

    for rank, (item_id, score) in enumerate(ranked_items, start=1):
        metadata = get_product_metadata(mapping_df, item_id)

        rows.append(
            {
                "RANK": rank,
                "ITEM_ID": metadata["ITEM_ID"],
                "PRODUCT_NAME": metadata["PRODUCT_NAME"],
                "CATEGORY": metadata["CATEGORY"],
                "SIMILARITY_SCORE": round(float(score), 4),
            }
        )

    return rows


def build_anchor_product_comparison(
    anchor_item_id: str,
    mapping_df: pl.DataFrame,
    product2vec_model: BaseModel,
    baseline_transformer_model: BaseModel,
    fine_tuned_transformer_model: BaseModel,
    k: int = 10,
) -> pl.DataFrame:
    """Compare all three models for the same anchor product."""
    anchor_metadata = get_product_metadata(mapping_df, anchor_item_id)

    models: list[tuple[str, BaseModel]] = [
        ("Product2Vec", product2vec_model),
        ("Baseline Sentence Transformer", baseline_transformer_model),
        ("Fine-Tuned Sentence Transformer", fine_tuned_transformer_model),
    ]

    rows: list[dict[str, Any]] = []

    for model_name, model in models:
        top_k_rows = get_top_k_by_cosine(
            model=model,
            anchor_item_id=anchor_item_id,
            mapping_df=mapping_df,
            k=k,
        )

        for row in top_k_rows:
            rows.append(
                {
                    "ANCHOR_ITEM_ID": anchor_metadata["ITEM_ID"],
                    "ANCHOR_PRODUCT_NAME": anchor_metadata["PRODUCT_NAME"],
                    "ANCHOR_CATEGORY": anchor_metadata["CATEGORY"],
                    "MODEL": model_name,
                    **row,
                }
            )

    return pl.DataFrame(rows)


def build_dissertation_comparison_table(
    comparison_df: pl.DataFrame,
) -> pl.DataFrame:
    """Create compact one-row-per-model table for dissertation reporting."""
    rows: list[dict[str, str]] = []

    for model_name in comparison_df["MODEL"].unique().to_list():
        model_df = comparison_df.filter(pl.col("MODEL") == model_name).sort("RANK")

        anchor_name = model_df["ANCHOR_PRODUCT_NAME"][0]

        retrieved = [
            f"{row['RANK']}. {row['PRODUCT_NAME']} ({row['SIMILARITY_SCORE']})"
            for row in model_df.iter_rows(named=True)
        ]

        rows.append(
            {
                "ANCHOR_PRODUCT_NAME": anchor_name,
                "MODEL": model_name,
                "TOP_RETRIEVED_PRODUCTS": "\n".join(retrieved),
            }
        )

    return pl.DataFrame(rows)
