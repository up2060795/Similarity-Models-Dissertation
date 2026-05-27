"""Data loading utilities for the similarity model pipeline.

Provides functions for loading basket and mapping data from S3,
with helpers for path construction, date extraction, and filtering.

All S3 reads use storage_options for credential injection so no
credentials are hardcoded or assumed from the environment.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import fsspec
import polars as pl

log = logging.getLogger(__name__)


def build_s3_path(bucket: str, key: str) -> str:
    """Build a full S3 path from bucket and key."""
    cleaned_bucket = bucket.rstrip("/")
    if not cleaned_bucket.startswith("s3://"):
        cleaned_bucket = f"s3://{cleaned_bucket}"

    cleaned_key = key.lstrip("/")
    return f"{cleaned_bucket}/{cleaned_key}"


def read_parquet_from_s3(
    path: str,
    storage_options: dict[str, Any],
) -> pl.DataFrame:
    """Read a single parquet file from S3."""
    with fsspec.open(path, "rb", **storage_options) as file_obj:
        return pl.read_parquet(file_obj)


def date_range(start_date: date, end_date: date) -> list[date]:
    """Return all dates in an inclusive date range."""
    if start_date > end_date:
        raise ValueError(
            f"start_date ({start_date}) must be on or before end_date ({end_date})."
        )

    days: list[date] = []
    current = start_date
    while current <= end_date:
        days.append(current)
        current += timedelta(days=1)
    return days


def build_daily_basket_keys(
    prefix: str,
    start_date: date,
    end_date: date,
    suffix: str = ".pq",
) -> list[str]:
    """Build candidate daily parquet keys for basket files.

    Assumes daily files are named YYYY-MM-DD.pq inside the given prefix, e.g.
    basket_data/2024-01-01.pq

    Adjust this function if your actual file naming pattern differs.
    """
    return [
        f"{prefix.strip('/')}/{day.strftime('%Y-%m-%d')}{suffix}"
        for day in date_range(start_date, end_date)
    ]


def _is_missing_file_error(err: Exception) -> bool:
    """Heuristically identify missing-file errors from S3/fsspec backends."""
    message = str(err).lower()
    missing_markers = [
        "no such key",
        "not found",
        "404",
        "does not exist",
        "the specified key does not exist",
        "path does not exist",
    ]
    return any(marker in message for marker in missing_markers)


def load_basket_data(
    bucket: str,
    prefix: str,
    start_date: date,
    end_date: date,
    storage_options: dict[str, Any],
    suffix: str = ".pq",
) -> pl.DataFrame:
    """Load basket data from S3 for an inclusive date range.

    Reads only expected daily files instead of scanning the whole prefix and
    filtering afterwards.
    """
    keys = build_daily_basket_keys(
        prefix=prefix,
        start_date=start_date,
        end_date=end_date,
        suffix=suffix,
    )

    dataframes: list[pl.DataFrame] = []
    missing_files: list[str] = []

    for key in keys:
        path = build_s3_path(bucket=bucket, key=key)
        log.info("Trying basket file: %s", path)
        try:
            df = read_parquet_from_s3(path=path, storage_options=storage_options)
            log.info("Loaded basket file: %s | rows=%s", path, f"{df.height:,}")
            dataframes.append(df)
        except Exception as err:
            if _is_missing_file_error(err):
                log.warning("Missing or unreadable expected basket file: %s", path)
                missing_files.append(path)
            else:
                log.exception("Unexpected error while reading %s", path)
                raise

    if not dataframes:
        raise ValueError(
            "No basket parquet files could be loaded for the requested date range. "
            f"Tried {len(keys)} files under prefix '{prefix}'."
        )

    combined = pl.concat(dataframes, how="vertical")

    if missing_files:
        log.warning(
            "%s expected daily files were missing or unreadable.",
            len(missing_files),
        )

    return combined


def load_mapping_data(
    bucket: str,
    key: str,
    storage_options: dict[str, Any],
) -> pl.DataFrame:
    """Load the item mapping table from a single parquet file in S3."""
    path = build_s3_path(bucket=bucket, key=key)
    log.info("Loading mapping data from %s", path)
    return read_parquet_from_s3(path=path, storage_options=storage_options)
