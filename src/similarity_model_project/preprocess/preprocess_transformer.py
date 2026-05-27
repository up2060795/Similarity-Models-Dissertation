from __future__ import annotations

from dataclasses import dataclass

import polars as pl


def build_transformer_products(
    mapping_df: pl.DataFrame,
    text_mode: str = "name_and_category",
) -> list[tuple[str, str]]:
    """Build (item_id, text) pairs for sentence-transformer encoding."""
    required_cols = {"ITEM_ID", "PRODUCT_NAME", "CATEGORY"}
    missing = required_cols.difference(mapping_df.columns)
    if missing:
        raise ValueError(
            f"Mapping dataframe missing required columns: {sorted(missing)}"
        )

    df = (
        mapping_df.select(["ITEM_ID", "PRODUCT_NAME", "CATEGORY"])
        .drop_nulls()
        .unique(subset=["ITEM_ID"])
    )

    product_pairs: list[tuple[str, str]] = []

    for row in df.iter_rows(named=True):
        item_id = row["ITEM_ID"]
        product_name = row["PRODUCT_NAME"]
        category = row["CATEGORY"]

        if text_mode == "name_only":
            text = str(product_name)
        elif text_mode == "name_and_category":
            text = f"{product_name}. Category: {category}."
        else:
            raise ValueError(f"Unsupported text_mode: {text_mode}")

        product_pairs.append((str(item_id), text))

    return product_pairs


@dataclass
class TransformerTrainingPairs:
    """Container for sentence transformer fine-tuning pairs."""

    positive_pairs: pl.DataFrame
    product_text_df: pl.DataFrame


def build_transformer_product_text_df(
    mapping_df: pl.DataFrame,
    text_mode: str = "name_and_category",
) -> pl.DataFrame:
    """Build product text dataframe for sentence-transformer fine-tuning."""
    product_pairs = build_transformer_products(
        mapping_df=mapping_df,
        text_mode=text_mode,
    )

    return pl.DataFrame(
        product_pairs,
        schema=["ITEM_ID", "PRODUCT_TEXT"],
        orient="row",
    )


def build_positive_transformer_pairs(
    cooccurrence_df: pl.DataFrame,
    product_text_df: pl.DataFrame,
    min_cooccurrence_count: int = 3,
    max_pairs: int | None = None,
) -> pl.DataFrame:
    """Build positive sentence-transformer training pairs from co-occurrence data."""
    required_cols = {"PRODUCT_A", "PRODUCT_B", "CO_OCCURRENCE_COUNT"}
    missing = required_cols.difference(cooccurrence_df.columns)

    if missing:
        raise ValueError(
            f"Cooccurrence dataframe missing required columns: {sorted(missing)}"
        )

    product_a_text = product_text_df.select(
        [
            pl.col("ITEM_ID").alias("PRODUCT_A"),
            pl.col("PRODUCT_TEXT").alias("TEXT_A"),
        ]
    )

    product_b_text = product_text_df.select(
        [
            pl.col("ITEM_ID").alias("PRODUCT_B"),
            pl.col("PRODUCT_TEXT").alias("TEXT_B"),
        ]
    )

    positive_pairs = (
        cooccurrence_df.filter(pl.col("CO_OCCURRENCE_COUNT") >= min_cooccurrence_count)
        .join(product_a_text, on="PRODUCT_A", how="inner")
        .join(product_b_text, on="PRODUCT_B", how="inner")
        .select(
            [
                "PRODUCT_A",
                "PRODUCT_B",
                "TEXT_A",
                "TEXT_B",
                "CO_OCCURRENCE_COUNT",
            ]
        )
        .sort("CO_OCCURRENCE_COUNT", descending=True)
    )

    if max_pairs is not None:
        positive_pairs = positive_pairs.head(max_pairs)

    return positive_pairs


def preprocess_transformer_training_pairs(
    mapping_df: pl.DataFrame,
    cooccurrence_df: pl.DataFrame,
    text_mode: str = "name_and_category",
    min_cooccurrence_count: int = 3,
    max_pairs: int | None = None,
) -> TransformerTrainingPairs:
    """Prepare product text and positive pairs for sentence-transformer fine-tuning."""
    product_text_df = build_transformer_product_text_df(
        mapping_df=mapping_df,
        text_mode=text_mode,
    )

    positive_pairs = build_positive_transformer_pairs(
        cooccurrence_df=cooccurrence_df,
        product_text_df=product_text_df,
        min_cooccurrence_count=min_cooccurrence_count,
        max_pairs=max_pairs,
    )

    return TransformerTrainingPairs(
        positive_pairs=positive_pairs,
        product_text_df=product_text_df,
    )


def build_labelled_transformer_pairs(
    positive_pairs_df: pl.DataFrame,
    product_text_df: pl.DataFrame,
    mapping_df: pl.DataFrame,
    negative_ratio: int = 1,
) -> pl.DataFrame:
    """Build labelled positive and hard-negative transformer pairs."""
    max_count = positive_pairs_df["CO_OCCURRENCE_COUNT"].max()

    positives = positive_pairs_df.with_columns(
        (pl.col("CO_OCCURRENCE_COUNT") / max_count).clip(0.1, 1.0).alias("LABEL")
    )

    product_meta = (
        mapping_df.select(["ITEM_ID", "CATEGORY"])
        .drop_nulls()
        .unique(subset=["ITEM_ID"])
    )

    positive_ids = set(
        zip(
            positive_pairs_df["PRODUCT_A"].to_list(),
            positive_pairs_df["PRODUCT_B"].to_list(),
        )
    )

    product_rows = product_text_df.join(product_meta, on="ITEM_ID", how="inner")

    negatives: list[dict[str, object]] = []

    for row in positives.iter_rows(named=True):
        product_a = row["PRODUCT_A"]
        category_a = product_rows.filter(pl.col("ITEM_ID") == product_a)[
            "CATEGORY"
        ].item()

        candidates = product_rows.filter(
            (pl.col("CATEGORY") != category_a) & (pl.col("ITEM_ID") != product_a)
        )

        if candidates.height == 0:
            continue

        sampled = candidates.sample(n=min(negative_ratio, candidates.height), seed=42)

        for neg in sampled.iter_rows(named=True):
            product_b = neg["ITEM_ID"]

            if (product_a, product_b) in positive_ids:
                continue

            negatives.append(
                {
                    "PRODUCT_A": product_a,
                    "PRODUCT_B": product_b,
                    "TEXT_A": row["TEXT_A"],
                    "TEXT_B": neg["PRODUCT_TEXT"],
                    "CO_OCCURRENCE_COUNT": 0,
                    "LABEL": 0.0,
                }
            )

    negatives_df = pl.DataFrame(negatives) if negatives else pl.DataFrame()

    if negatives_df.height == 0:
        return positives

    return pl.concat(
        [
            positives.select(
                [
                    "PRODUCT_A",
                    "PRODUCT_B",
                    "TEXT_A",
                    "TEXT_B",
                    "CO_OCCURRENCE_COUNT",
                    "LABEL",
                ]
            ),
            negatives_df,
        ],
        how="vertical",
    )
