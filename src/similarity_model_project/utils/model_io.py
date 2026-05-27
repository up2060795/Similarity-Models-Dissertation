"""Model and artifact I/O helpers."""

# ruff: noqa: D103, ANN001, ANN201
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import fsspec
import polars as pl

log = logging.getLogger(__name__)


def upload_local_file_to_s3(
    local_path: Path,
    s3_path: str,
    storage_options: dict[str, Any],
) -> None:
    """Upload a local file to an S3 path using streamed chunks."""
    chunk_size = 1024 * 1024
    log.info("Uploading %s to %s", local_path, s3_path)

    with local_path.open("rb") as local_file:
        with fsspec.open(s3_path, "wb", **storage_options) as remote_file:
            while chunk := local_file.read(chunk_size):
                remote_file.write(chunk)

    log.info("Completed upload of %s to %s", local_path, s3_path)


def download_s3_file_to_local(
    s3_path: str,
    local_path: Path,
    storage_options: dict[str, Any],
) -> Path:
    """Download a file from S3 to a local path."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = 1024 * 1024

    log.info("Downloading %s to %s", s3_path, local_path)

    with fsspec.open(s3_path, "rb", **storage_options) as remote_file:
        with local_path.open("wb") as local_file:
            while chunk := remote_file.read(chunk_size):
                local_file.write(chunk)

    log.info("Completed download of %s to %s", s3_path, local_path)
    return local_path


def save_model_locally(
    model,
    local_path: Path,
) -> Path:
    """Serialize and save a trained model locally."""
    local_path.parent.mkdir(parents=True, exist_ok=True)

    with local_path.open("wb") as file_obj:
        pickle.dump(model, file_obj)

    log.info("Model saved locally: %s", local_path)
    return local_path


def load_model_from_local(
    local_path: Path,
):
    """Load a serialized model from local storage."""
    if not local_path.exists():
        raise FileNotFoundError(f"Local model file not found: {local_path}")

    with local_path.open("rb") as file_obj:
        model = pickle.load(file_obj)

    log.info("Model loaded from local path: %s", local_path)
    return model


def build_s3_model_path(
    s3_bucket: str,
    s3_results_prefix: str,
    train_start_date,
    train_end_date,
    dataset_hash: str,
) -> str:
    """Construct the canonical S3 path for a trained model."""
    return (
        f"s3://{s3_bucket}/{s3_results_prefix}/"
        f"prod2vec_{train_start_date}_{train_end_date}_{dataset_hash}.pkl"
    )


def save_model_to_s3(
    model_path: Path,
    s3_bucket: str,
    s3_results_prefix: str,
    train_start_date,
    train_end_date,
    dataset_hash: str,
    storage_options: dict[str, Any],
) -> str:
    """Save trained model file to S3."""
    s3_model_path = build_s3_model_path(
        s3_bucket=s3_bucket,
        s3_results_prefix=s3_results_prefix,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        dataset_hash=dataset_hash,
    )
    try:
        upload_local_file_to_s3(model_path, s3_model_path, storage_options)
        log.info("Model saved to S3: %s", s3_model_path)
    except Exception as err:
        log.warning("Failed to save model to S3: %s", err)
        raise

    return s3_model_path


def load_model_from_s3(
    s3_bucket: str,
    s3_results_prefix: str,
    train_start_date,
    train_end_date,
    dataset_hash: str,
    local_model_dir: Path,
    storage_options: dict[str, Any],
):
    """Download a trained model from S3 and load it into memory."""
    s3_model_path = build_s3_model_path(
        s3_bucket=s3_bucket,
        s3_results_prefix=s3_results_prefix,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        dataset_hash=dataset_hash,
    )

    local_model_path = local_model_dir / (
        f"prod2vec_{train_start_date}_{train_end_date}_{dataset_hash}.pkl"
    )

    download_s3_file_to_local(
        s3_path=s3_model_path,
        local_path=local_model_path,
        storage_options=storage_options,
    )

    model = load_model_from_local(local_model_path)
    return model, local_model_path, s3_model_path


def save_embeddings(
    model,
    mapping_df: pl.DataFrame,
    config,
    storage_options: dict[str, Any],
) -> Path:
    """Save learned product embeddings to CSV and Parquet."""
    embeddings = model.get_all_embeddings()
    if not embeddings:
        raise ValueError("Model returned no embeddings to save.")

    log.info("=" * 70)
    log.info("SAVING EMBEDDINGS")
    log.info("=" * 70)

    item_ids = list(embeddings.keys())
    vectors = [embeddings[item_id] for item_id in item_ids]

    dim = len(vectors[0])
    embedding_cols = [f"EMBEDDING_{i}" for i in range(dim)]

    df = pl.DataFrame(
        [vector.tolist() for vector in vectors],
        schema=embedding_cols,
    ).with_columns(pl.Series("ITEM_ID", item_ids))

    mapping_small = mapping_df.select(["ITEM_ID", "PRODUCT_NAME", "CATEGORY"])
    df = df.join(mapping_small, on="ITEM_ID", how="left")
    df = df.select(["ITEM_ID", "PRODUCT_NAME", "CATEGORY"] + embedding_cols)

    model_label = getattr(config, "pipeline_mode", "model")

    local_csv_path = (
        config.local_results_dir
        / f"embeddings_{model_label}_{config.test_start_date}_{config.test_end_date}.csv"
    )
    local_parquet_path = (
        config.local_results_dir
        / f"embeddings_{model_label}_{config.test_start_date}_{config.test_end_date}.pq"
    )

    df.write_csv(local_csv_path)
    df.write_parquet(local_parquet_path)

    log.info("Embeddings saved locally: %s", local_csv_path)
    log.info("Embeddings parquet saved locally: %s", local_parquet_path)

    s3_parquet_path = (
        f"s3://{config.s3_bucket}/{config.s3_results_prefix}/"
        f"embeddings_{model_label}_{config.test_start_date}_{config.test_end_date}.pq"
    )

    try:
        upload_local_file_to_s3(
            local_path=local_parquet_path,
            s3_path=s3_parquet_path,
            storage_options=storage_options,
        )
        log.info("Embeddings saved to S3: %s", s3_parquet_path)
    except Exception as err:
        log.warning("Could not save embeddings to S3: %s", err)

    return local_csv_path


def save_run_metadata(
    config,
    raw_df: pl.DataFrame,
    result,
    model,
    eval_df: pl.DataFrame,
    model_path: Path,
    storage_options: dict[str, Any],
) -> Path:
    """Save metadata describing the pipeline run."""
    embeddings = model.get_all_embeddings()
    if not embeddings:
        raise ValueError("Model returned no embeddings to save.")

    def mean_or_none(df: pl.DataFrame, column: str) -> float | None:
        if column not in df.columns:
            return None
        valid = df[column].drop_nulls()
        if len(valid) == 0:
            return None
        return float(valid.mean())

    metadata = {
        "train_start_date": config.train_start_date,
        "train_end_date": config.train_end_date,
        "test_start_date": config.test_start_date,
        "test_end_date": config.test_end_date,
        "raw_rows_loaded": raw_df.height,
        "processed_baskets": result.dataset.height,
        "dataset_hash": result.dataset_hash,
        "processed_output_path": str(result.output_path),
        "model_output_path": str(model_path),
        "vocabulary_size": len(model.model_.wv),
        "products_evaluated": eval_df.height,
        "mean_test_precision_at_3": mean_or_none(eval_df, "TEST_PRECISION_AT_3"),
        "mean_test_precision_at_5": mean_or_none(eval_df, "TEST_PRECISION_AT_5"),
        "mean_test_precision_at_10": mean_or_none(eval_df, "TEST_PRECISION_AT_10"),
        "mean_category_precision_at_3": mean_or_none(
            eval_df, "CATEGORY_PRECISION_AT_3"
        ),
        "mean_category_precision_at_5": mean_or_none(
            eval_df, "CATEGORY_PRECISION_AT_5"
        ),
        "mean_category_precision_at_10": mean_or_none(
            eval_df, "CATEGORY_PRECISION_AT_10"
        ),
    }

    metadata_path = config.local_results_dir / (
        f"run_metadata_"
        f"{config.train_start_date}_{config.train_end_date}_"
        f"{result.dataset_hash}.json"
    )
    with metadata_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2, default=str)

    log.info("Run metadata saved locally: %s", metadata_path)

    s3_metadata_path = (
        f"s3://{config.s3_bucket}/{config.s3_results_prefix}/"
        f"run_metadata_{config.train_start_date}_{config.train_end_date}_{result.dataset_hash}.json"
    )
    try:
        upload_local_file_to_s3(metadata_path, s3_metadata_path, storage_options)
        log.info("Run metadata saved to S3: %s", s3_metadata_path)
    except Exception as err:
        log.warning(
            "Could not upload metadata to S3. Local copy remains at %s. Error: %s",
            metadata_path,
            err,
        )

    return metadata_path
