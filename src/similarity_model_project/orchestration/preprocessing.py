# ruff: noqa: D103, ANN001
from __future__ import annotations

import logging
from typing import Any

import polars as pl

from similarity_model_project.preprocess.preprocess_product2vec import (
    ITEM_COUNT_COL,
    Prod2VecDatasetResult,
    compute_dataset_hash,
    validate_aggregated_basket_df,
)
from similarity_model_project.utils.model_io import upload_local_file_to_s3

log = logging.getLogger(__name__)


def phase_2_preprocess(
    raw_df: pl.DataFrame,
    config,
    storage_options: dict[str, Any],
    dataset_label: str,
) -> Prod2VecDatasetResult:
    log.info("=" * 70)
    log.info("PHASE 2 — DATA PREPROCESSING (%s)", dataset_label.upper())
    log.info("Min basket size      : %s", config.min_basket_size)
    log.info("Deduplicate items    : %s", config.deduplicate_items)
    log.info("=" * 70)

    validate_aggregated_basket_df(raw_df)

    processed_df = raw_df.filter(pl.col(ITEM_COUNT_COL) >= config.min_basket_size)

    if processed_df.height == 0:
        raise ValueError(
            f"No baskets remain after filtering {dataset_label} "
            "dataset by minimum basket size."
        )

    if config.deduplicate_items:
        processed_df = (
            processed_df.with_columns(pl.col(config.item_id_col).list.unique())
            .with_columns(pl.col(config.item_id_col).list.len().alias(ITEM_COUNT_COL))
            .filter(pl.col(ITEM_COUNT_COL) >= config.min_basket_size)
        )

    dataset_hash = compute_dataset_hash(processed_df)

    output_path = (
        config.local_processed_dir
        / f"{dataset_label}_dataset_prod2vec_{dataset_hash}.pq"
    )

    processed_df.write_parquet(output_path)

    result = Prod2VecDatasetResult(
        dataset=processed_df,
        dataset_hash=dataset_hash,
        output_path=output_path,
    )

    log.info(
        "%s baskets retained : %s",
        dataset_label.capitalize(),
        f"{result.dataset.height:,}",
    )
    log.info("Dataset hash         : %s", result.dataset_hash)
    log.info("Processed file       : %s", result.output_path)

    log.info(
        "Basket stats         : min=%s | max=%s | mean=%.2f",
        result.dataset[ITEM_COUNT_COL].min(),
        result.dataset[ITEM_COUNT_COL].max(),
        result.dataset[ITEM_COUNT_COL].mean(),
    )

    s3_processed_path = (
        f"s3://{config.s3_bucket}/"
        f"{config.s3_results_prefix}/"
        f"{dataset_label}_processed_{result.dataset_hash}.pq"
    )

    try:
        upload_local_file_to_s3(
            local_path=result.output_path,
            s3_path=s3_processed_path,
            storage_options=storage_options,
        )

        log.info(
            "%s dataset saved to S3: %s",
            dataset_label.capitalize(),
            s3_processed_path,
        )

    except Exception as err:
        log.warning(
            "Could not save %s dataset to S3. Local copy remains at %s. Error: %s",
            dataset_label,
            result.output_path,
            err,
        )

    return result
