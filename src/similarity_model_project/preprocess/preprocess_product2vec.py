"""Preprocessing pipeline for Product2Vec training and evaluation data.

This module handles two different stages of data preparation:

    STAGE 1 — Raw transaction data (Format A)
    -----------------------------------------
    Input:  One row per basket-item pair, e.g.:
                BASKET_ID  | ITEM_ID
                BASKET_001 | ITEM_A
                BASKET_001 | ITEM_B
                BASKET_002 | ITEM_C

    Pipeline:
        1. Load raw data from parquet or CSV
        2. Validate required columns and data quality
        3. Group items into per-basket lists
        4. Filter out baskets that are too small to be useful
        5. Save the processed dataset with a content hash for reproducibility

    STAGE 2 — Pre-aggregated basket data (Format B)
    ------------------------------------------------
    Input:  One row per basket, already grouped, loaded from S3, e.g.:
                ITEM_ID                   | ITEM_COUNT
                ["ITEM_A", "ITEM_B", ...] | 3

    This format is produced by the daily basket composition pipeline and
    loaded directly into the similarity orchestration pipeline. It requires
    a separate validation function (validate_aggregated_basket_df) and a
    direct converter (baskets_for_prod2vec) rather than the full pipeline.

    STAGE 3 — Held-out evaluation ground truth
    ------------------------------------------
    Input:  Processed basket data in Format B
    Output: Per-product relevance sets built from basket co-occurrence,
            used for held-out test evaluation of Product2Vec recommendations.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import polars as pl

# ── Constants ─────────────────────────────────────────────────────────────────

# Column names for the pre-aggregated Format B schema (loaded from S3).
# Defined as constants so any rename only needs to happen in one place.
ITEM_ID_COL: str = "ITEM_ID"
ITEM_COUNT_COL: str = "ITEM_COUNT"


# ── Output container ──────────────────────────────────────────────────────────


@dataclass
class Prod2VecDatasetResult:
    """Container returned by the preprocessing pipeline.

    Bundles the processed DataFrame together with metadata so the caller
    has everything needed to train a model and track which data was used.

    Attributes:
        dataset:      Processed DataFrame with basket-level grouped items.
                      ITEM_ID is a List[Utf8] column — one list per basket.
        dataset_hash: Short 16-character SHA256 hash of the dataset content.
                      Used for cache-busting and tracking which data a model
                      was trained on — same data always produces the same hash.
        output_path:  Path where the processed parquet file was saved to disk.
    """

    dataset: pl.DataFrame
    dataset_hash: str
    output_path: Path


# ── Step 1: Load raw data ─────────────────────────────────────────────────────


def load_data(path: str | Path) -> pl.DataFrame:
    """Load raw transaction data from a parquet (.pq / .parquet) or CSV file.

    Args:
        path: Path to the input file. Accepts both string and Path objects.

    Returns:
        Raw DataFrame exactly as stored on disk — no transformations applied.

    Raises:
        FileNotFoundError: If no file exists at the given path.
        ValueError: If the file extension is not .parquet, .pq, or .csv.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)

    if suffix == ".csv":
        return pl.read_csv(path)

    raise ValueError(
        f"Unsupported file type: '{suffix}'. Expected .parquet, .pq, or .csv."
    )


# ── Step 2: Validate raw input (Format A) ────────────────────────────────────


def validate_basket_data(
    df: pl.DataFrame,
    basket_id_col: str = "BASKET_ID",
    item_id_col: str = "ITEM_ID",
) -> None:
    """Validate that raw transaction data is safe to process.

    Checks performed:
        1. Required columns are present
        2. DataFrame is not empty
        3. No null values in BASKET_ID
        4. No null values in ITEM_ID

    Args:
        df: Raw transaction DataFrame to validate.
        basket_id_col: Name of the column identifying each basket.
        item_id_col: Name of the column identifying each item.

    Raises:
        ValueError: If any validation check fails.
    """
    required = [basket_id_col, item_id_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.height == 0:
        raise ValueError("Input DataFrame is empty.")

    if df.select(pl.col(basket_id_col).is_null().any()).item():
        raise ValueError(f"Column '{basket_id_col}' contains null values.")

    if df.select(pl.col(item_id_col).is_null().any()).item():
        raise ValueError(f"Column '{item_id_col}' contains null values.")


# ── Step 3: Build basket sequences (Format A → Format B) ─────────────────────


def preprocess_prod2vec_data(
    df: pl.DataFrame,
    basket_id_col: str = "BASKET_ID",
    item_id_col: str = "ITEM_ID",
    min_items_per_basket: int = 3,
    deduplicate_within_basket: bool = False,
) -> pl.DataFrame:
    """Convert raw transaction rows into Product2Vec basket sequences.

    Transforms Format A (one row per basket-item pair) into Format B
    (one row per basket with items grouped into a list), then filters
    out baskets that are too small to produce meaningful skip-grams.

    Args:
        df: Raw transaction DataFrame (Format A).
        basket_id_col: Column identifying each basket.
        item_id_col: Column identifying each item.
        min_items_per_basket: Baskets smaller than this are dropped.
        deduplicate_within_basket: If True, remove duplicate item IDs
            within the same basket before grouping.

    Returns:
        Processed DataFrame with columns:
            BASKET_ID: str
            ITEM_ID: List[Utf8]
            ITEM_COUNT: Int

    Raises:
        ValueError: If validation fails or no baskets remain after filtering.
    """
    if min_items_per_basket < 2:
        raise ValueError("min_items_per_basket must be at least 2.")

    validate_basket_data(df, basket_id_col=basket_id_col, item_id_col=item_id_col)

    working_df = (
        df.select([basket_id_col, item_id_col])
        .drop_nulls()
        .with_columns(
            pl.col(basket_id_col).cast(pl.String),
            pl.col(item_id_col).cast(pl.String),
        )
    )

    if deduplicate_within_basket:
        working_df = working_df.unique(subset=[basket_id_col, item_id_col])

    basket_df = (
        working_df.group_by(basket_id_col, maintain_order=True)
        .agg(pl.col(item_id_col))
        .rename({item_id_col: ITEM_ID_COL})
        .with_columns(pl.col(ITEM_ID_COL).list.len().alias(ITEM_COUNT_COL))
        .filter(pl.col(ITEM_COUNT_COL) >= min_items_per_basket)
        .select([basket_id_col, ITEM_ID_COL, ITEM_COUNT_COL])
    )

    if basket_df.height == 0:
        raise ValueError(
            "No baskets remain after preprocessing. "
            "Check your raw data or reduce min_items_per_basket."
        )

    return basket_df


# ── Step 4: Hash the dataset ──────────────────────────────────────────────────


def compute_dataset_hash(df: pl.DataFrame) -> str:
    """Compute a short stable content hash for a processed basket DataFrame.

    Args:
        df: Processed basket DataFrame to hash.

    Returns:
        16-character hexadecimal string.
    """
    buffer = io.BytesIO()
    df.write_ipc(buffer)
    raw_bytes = buffer.getvalue()
    return hashlib.sha256(raw_bytes).hexdigest()[:16]


# ── Step 5: Save processed dataset ───────────────────────────────────────────


def save_prod2vec_dataset(
    df: pl.DataFrame,
    output_dir: str | Path,
    prefix: str = "training_dataset_prod2vec",
) -> tuple[Path, str]:
    """Save the processed basket DataFrame as a parquet file.

    Args:
        df: Processed basket DataFrame to save.
        output_dir: Directory to write the file into.
        prefix: Filename prefix.

    Returns:
        Tuple of (output_path, dataset_hash).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_hash = compute_dataset_hash(df)
    output_path = output_dir / f"{prefix}_{dataset_hash}.pq"
    df.write_parquet(output_path)

    return output_path, dataset_hash


# ── Full Format A pipeline ────────────────────────────────────────────────────


def prepare_prod2vec_training_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    basket_id_col: str = "BASKET_ID",
    item_id_col: str = "ITEM_ID",
    min_items_per_basket: int = 3,
    deduplicate_within_basket: bool = False,
) -> Prod2VecDatasetResult:
    """End-to-end preprocessing pipeline for Product2Vec training data.

    Args:
        input_path: Path to raw parquet or CSV file.
        output_dir: Directory to write the processed dataset to.
        basket_id_col: Column name for basket IDs.
        item_id_col: Column name for item IDs.
        min_items_per_basket: Minimum items to keep a basket.
        deduplicate_within_basket: Remove duplicate items per basket.

    Returns:
        Prod2VecDatasetResult containing the processed dataset and metadata.
    """
    raw_df = load_data(input_path)

    processed_df = preprocess_prod2vec_data(
        df=raw_df,
        basket_id_col=basket_id_col,
        item_id_col=item_id_col,
        min_items_per_basket=min_items_per_basket,
        deduplicate_within_basket=deduplicate_within_basket,
    )

    output_path, dataset_hash = save_prod2vec_dataset(
        df=processed_df,
        output_dir=output_dir,
        prefix="training_dataset_prod2vec",
    )

    return Prod2VecDatasetResult(
        dataset=processed_df,
        dataset_hash=dataset_hash,
        output_path=output_path,
    )


# ── Format B converter ────────────────────────────────────────────────────────


def baskets_for_prod2vec(
    df: pl.DataFrame,
    item_list_col: str = ITEM_ID_COL,
) -> list[list[str]]:
    """Convert a processed basket DataFrame into Product2Vec model input.

    Args:
        df: Processed basket DataFrame.
        item_list_col: Name of the list column.

    Returns:
        list[list[str]] ready to pass directly to Product2Vec.fit().

    Raises:
        ValueError: If item_list_col is not found in the DataFrame.
    """
    if item_list_col not in df.columns:
        raise ValueError(
            f"Column '{item_list_col}' not found in DataFrame. "
            f"Available columns: {df.columns}"
        )

    return df[item_list_col].to_list()


# ── Format B validator ────────────────────────────────────────────────────────


def validate_aggregated_basket_df(df: pl.DataFrame) -> None:
    """Validate a pre-aggregated basket DataFrame loaded from S3 (Format B).

    Expected schema:
        ITEM_ID: List[Utf8]
        ITEM_COUNT: Int

    Args:
        df: Pre-aggregated basket DataFrame to validate.

    Raises:
        TypeError: If df is not a Polars DataFrame.
        ValueError: If required columns are missing, have wrong dtypes,
            or the DataFrame is empty.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(
            f"Expected a Polars DataFrame, got {type(df).__name__}. "
            "If passing a Pandas DataFrame, convert with pl.from_pandas() first."
        )

    if df.height == 0:
        raise ValueError(
            "DataFrame is empty — no baskets to train on. "
            "Check that the S3 source data exists for the configured date range."
        )

    missing = {ITEM_ID_COL, ITEM_COUNT_COL} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    id_dtype = df[ITEM_ID_COL].dtype
    if id_dtype not in (pl.List(pl.Utf8), pl.List(pl.String)):
        raise ValueError(
            f"'{ITEM_ID_COL}' must be List(Utf8), got {id_dtype}. "
            "Run preprocess_prod2vec_data() first to produce the correct schema."
        )

    count_dtype = df[ITEM_COUNT_COL].dtype
    if count_dtype not in (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    ):
        raise ValueError(
            f"'{ITEM_COUNT_COL}' must be an integer type, got {count_dtype}."
        )


# ── Held-out evaluation helper ───────────────────────────────────────────────


def build_cooccurrence_counts(
    df: pl.DataFrame,
    item_list_col: str = ITEM_ID_COL,
) -> dict[str, dict[str, int]]:
    """Build raw basket co-occurrence counts for every product pair.

    Returns:
        Nested dict: focal ITEM_ID -> {related ITEM_ID -> count}.
    """
    validate_aggregated_basket_df(df)

    if item_list_col not in df.columns:
        raise ValueError(
            f"Column '{item_list_col}' not found in DataFrame. "
            f"Available columns: {df.columns}"
        )

    cooccurrence_counts: dict[str, dict[str, int]] = {}

    for basket in df[item_list_col].to_list():
        unique_items = list(dict.fromkeys(basket))

        if len(unique_items) < 2:
            continue

        for focal_item in unique_items:
            if focal_item not in cooccurrence_counts:
                cooccurrence_counts[focal_item] = {}

            for related_item in unique_items:
                if related_item == focal_item:
                    continue

                if related_item not in cooccurrence_counts[focal_item]:
                    cooccurrence_counts[focal_item][related_item] = 0

                cooccurrence_counts[focal_item][related_item] += 1

    return cooccurrence_counts


def build_cooccurrence_ground_truth(
    df: pl.DataFrame,
    item_list_col: str = ITEM_ID_COL,
    top_n: int = 50,
) -> dict[str, set[str]]:
    """Build held-out relevance sets from top-N basket co-occurrence.

    For each focal product, relevant products are defined as the top-N most
    frequently co-occurring products observed in the same baskets.

    Example:
        Basket 1: [A, B, C]
        Basket 2: [A, D]
        Basket 3: [A, B]

        Output with top_n=2:
            A -> {B, C}
            B -> {A, C}
            C -> {A, B}
            D -> {A}

    Args:
        df: Processed basket DataFrame in Format B.
        item_list_col: Name of the basket item-list column.
        top_n: Number of most frequent co-occurring items to keep per focal
            product.

    Returns:
        Dictionary mapping focal ITEM_ID -> set of relevant ITEM_IDs.

    Raises:
        ValueError: If the dataframe does not contain the expected item-list
            column, or if top_n is not positive.
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}.")

    cooccurrence_counts = build_cooccurrence_counts(df, item_list_col)

    ground_truth: dict[str, set[str]] = {}

    for focal_item, related_counts in cooccurrence_counts.items():
        ranked_related = sorted(
            related_counts.items(),
            key=lambda x: (-x[1], x[0]),
        )[:top_n]

        ground_truth[focal_item] = {item_id for item_id, _ in ranked_related}

    return ground_truth
