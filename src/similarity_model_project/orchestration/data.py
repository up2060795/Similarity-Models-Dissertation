# ruff: noqa: D103, ANN001
from __future__ import annotations

import logging
from typing import Any

import polars as pl

from similarity_model_project.preprocess.load_data import (
    load_basket_data,
    load_mapping_data,
)

log = logging.getLogger(__name__)


def cache_limit_label(config) -> str:
    return (
        "all"
        if config.max_baskets_for_dev is None
        else f"limit_{config.max_baskets_for_dev}"
    )


def validate_raw_basket_df(
    df: pl.DataFrame,
    basket_id_col: str,
    item_id_col: str,
) -> None:
    required_cols = [basket_id_col, item_id_col]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns in raw basket data: {missing_cols}")

    if df.height == 0:
        raise ValueError("Raw basket data is empty.")

    null_basket_ids = df.select(pl.col(basket_id_col).is_null().sum()).item()
    null_item_ids = df.select(pl.col(item_id_col).is_null().sum()).item()

    log.info("Raw data validation:")
    log.info("  Rows                       : %s", f"{df.height:,}")
    log.info("  Columns                    : %s", df.columns)
    log.info("  Null %s count             : %s", basket_id_col, f"{null_basket_ids:,}")
    log.info("  Null %s count             : %s", item_id_col, f"{null_item_ids:,}")
    log.info("  Schema                     : %s", df.schema)

    if null_basket_ids > 0 or null_item_ids > 0:
        raise ValueError(
            "Raw basket data contains null values in key columns. "
            "Fix this before preprocessing."
        )


def standardise_mapping_df(
    mapping_df: pl.DataFrame,
    item_id_col: str,
    category_col: str,
) -> pl.DataFrame:
    product_name_col = "ITEM_DESC"

    if item_id_col not in mapping_df.columns:
        raise ValueError(f"Mapping dataframe must contain {item_id_col}.")

    if category_col not in mapping_df.columns:
        raise ValueError(
            f"Mapping dataframe must contain configured category column: {category_col}"
        )

    if product_name_col not in mapping_df.columns:
        raise ValueError(
            f"Mapping dataframe must contain product name column: {product_name_col}"
        )

    return mapping_df.select(
        [
            pl.col(item_id_col).alias("ITEM_ID"),
            pl.col(product_name_col).alias("PRODUCT_NAME"),
            pl.col(category_col).alias("CATEGORY"),
        ]
    )


def load_standardised_mapping(
    config,
    storage_options: dict[str, Any],
) -> pl.DataFrame:
    mapping_df = load_mapping_data(
        bucket=config.s3_bucket,
        key=config.s3_mapping_key,
        storage_options=storage_options,
    )

    return standardise_mapping_df(
        mapping_df=mapping_df,
        item_id_col=config.item_id_col,
        category_col=config.category_col,
    )


def phase_1_load_train_test_data(
    config,
    storage_options: dict[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    log.info("=" * 70)
    log.info("PHASE 1 — DATA UNDERSTANDING & LOADING")
    log.info(
        "Train date range: %s → %s", config.train_start_date, config.train_end_date
    )
    log.info("Test date range : %s → %s", config.test_start_date, config.test_end_date)
    log.info("S3 source       : %s/%s", config.s3_bucket, config.s3_basket_prefix)
    log.info("=" * 70)

    config.local_data_dir.mkdir(parents=True, exist_ok=True)

    limit_label = cache_limit_label(config)

    train_cache_path = (
        config.local_data_dir
        / f"raw_train_baskets_{config.train_start_date}_{config.train_end_date}_{limit_label}.pq"
    )

    test_cache_path = (
        config.local_data_dir
        / f"raw_test_baskets_{config.test_start_date}_{config.test_end_date}_{limit_label}.pq"
    )

    use_cache = getattr(config, "use_cached_data", True)

    if use_cache and train_cache_path.exists() and test_cache_path.exists():
        log.info("Loading cached raw train dataset: %s", train_cache_path)
        train_df = pl.read_parquet(train_cache_path)

        log.info("Loading cached raw test dataset : %s", test_cache_path)
        test_df = pl.read_parquet(test_cache_path)

    else:
        log.info("Cached raw datasets not found. Loading from S3.")

        train_df = load_basket_data(
            bucket=config.s3_bucket,
            prefix=config.s3_basket_prefix,
            start_date=config.train_start_date,
            end_date=config.train_end_date,
            storage_options=storage_options,
        )

        test_df = load_basket_data(
            bucket=config.s3_bucket,
            prefix=config.s3_basket_prefix,
            start_date=config.test_start_date,
            end_date=config.test_end_date,
            storage_options=storage_options,
        )

        if config.max_baskets_for_dev is not None:
            if train_df.height > config.max_baskets_for_dev:
                train_df = train_df.sort(config.basket_id_col).head(
                    config.max_baskets_for_dev
                )
                log.info(
                    "Dev mode: truncated train data to %s rows",
                    f"{train_df.height:,}",
                )

            if test_df.height > config.max_baskets_for_dev:
                test_df = test_df.sort(config.basket_id_col).head(
                    config.max_baskets_for_dev
                )
                log.info(
                    "Dev mode: truncated test data to %s rows",
                    f"{test_df.height:,}",
                )

        train_df.write_parquet(train_cache_path)
        test_df.write_parquet(test_cache_path)

        log.info("Raw train dataset saved locally: %s", train_cache_path)
        log.info("Raw test dataset saved locally : %s", test_cache_path)

    validate_raw_basket_df(
        df=train_df,
        basket_id_col=config.basket_id_col,
        item_id_col=config.item_id_col,
    )

    validate_raw_basket_df(
        df=test_df,
        basket_id_col=config.basket_id_col,
        item_id_col=config.item_id_col,
    )

    return train_df, test_df
