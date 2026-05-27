from __future__ import annotations

import polars as pl

# ── Step 1: Validate incoming basket ─────────────────────────────────────────


def validate_basket(
    df: pl.DataFrame,
    basket_id_col: str = "BASKET_ID",
    item_id_col: str = "ITEM_ID",
    item_count_col: str = "ITEM_COUNT",
) -> None:
    """Validate a basket DataFrame has required columns and no nulls.

    Raises ValueError if invalid — fail fast before any processing.
    """
    required = [basket_id_col, item_id_col, item_count_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for col in [basket_id_col, item_id_col]:
        if df[col].null_count() > 0:
            raise ValueError(f"Null values found in column: {col}")


# ── Step 2: Explode + join (same as training but no skip-gram) ────────────────


def get_basket_items_with_descriptions(
    df: pl.DataFrame,
    mapping_df: pl.DataFrame,
    item_id_col: str = "ITEM_ID",
) -> pl.DataFrame:
    """Explode basket item list and join descriptions.

    Input:  raw basket DataFrame (one row per basket, ITEM_ID as list)
    Output: one row per item with ITEM_DESC joined.
    """
    return df.explode(item_id_col).join(
        mapping_df.select([item_id_col, "ITEM_DESC"]),
        on=item_id_col,
        how="left",
    )


# ── Step 3: Handle unseen items ───────────────────────────────────────────────


def flag_unseen_items(
    df: pl.DataFrame,
    description_col: str = "ITEM_DESC",
    flag_col: str = "IS_UNSEEN",
) -> pl.DataFrame:
    """Flag items that had no match in the mapping table (null description).

    These are items the model has never seen during training.
    """
    return df.with_columns(pl.col(description_col).is_null().alias(flag_col))


def drop_unseen_items(
    df: pl.DataFrame,
    flag_col: str = "IS_UNSEEN",
) -> pl.DataFrame:
    """Remove items flagged as unseen before passing to model."""
    return df.filter(pl.col(flag_col).eq(False)).drop(flag_col)


# ── Step 4: Build basket sentence ─────────────────────────────────────────────


def build_basket_sentence(
    df: pl.DataFrame,
    basket_id_col: str = "BASKET_ID",
    description_col: str = "ITEM_DESC",
    separator: str = " ",
) -> pl.DataFrame:
    """Aggregate item descriptions into a single string per basket.

    This is the 'sentence' that gets passed to the sentence transformer.

    Input:  one row per item with ITEM_DESC
    Output: one row per basket with BASKET_SENTENCE
    e.g. "Diet Coke Walkers Crisps San Pellegrino Water"
    """
    return (
        df.group_by(basket_id_col)
        .agg(pl.col(description_col).drop_nulls().alias("items"))
        .with_columns(pl.col("items").list.join(separator).alias("BASKET_SENTENCE"))
        .drop("items")
    )


# ── Step 5: Filter empty baskets after processing ────────────────────────────


def filter_empty_sentences(
    df: pl.DataFrame,
    sentence_col: str = "BASKET_SENTENCE",
) -> pl.DataFrame:
    """Remove baskets with empty sentences after dropping unseen items.

    Can't embed an empty string.
    """
    return df.filter(pl.col(sentence_col).is_not_null() & (pl.col(sentence_col) != ""))


# ── Pipeline ──────────────────────────────────────────────────────────────────


def preprocess_for_inference(
    basket_df: pl.DataFrame,
    mapping_df: pl.DataFrame,
) -> pl.DataFrame:
    """Full inference preprocessing pipeline.

    Steps:
        1. Validate input basket
        2. Explode items and join descriptions
        3. Flag and drop unseen items
        4. Build one sentence string per basket
        5. Filter any empty sentences

    Input:  raw basket DataFrame, item mapping DataFrame
    Output: DataFrame with BASKET_ID and BASKET_SENTENCE columns,
            ready to pass to sentence transformer .encode()
    """
    validate_basket(basket_df)

    return (
        basket_df.pipe(get_basket_items_with_descriptions, mapping_df)
        .pipe(flag_unseen_items)
        .pipe(drop_unseen_items)
        .pipe(build_basket_sentence)
        .pipe(filter_empty_sentences)
    )
