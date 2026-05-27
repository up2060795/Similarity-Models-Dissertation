# ruff: noqa: D103, ANN001
from __future__ import annotations

import os
from typing import Any


def ensure_dirs(config) -> None:
    config.local_data_dir.mkdir(parents=True, exist_ok=True)
    config.local_processed_dir.mkdir(parents=True, exist_ok=True)
    config.local_results_dir.mkdir(parents=True, exist_ok=True)


def validate_temporal_split(config) -> None:
    if config.train_start_date > config.train_end_date:
        raise ValueError("train_start_date must be on or before train_end_date.")
    if config.test_start_date > config.test_end_date:
        raise ValueError("test_start_date must be on or before test_end_date.")
    if config.train_end_date >= config.test_start_date:
        raise ValueError("Training period must end before test period begins.")


def get_storage_options() -> dict[str, Any]:
    options: dict[str, Any] = {}

    if key_id := os.environ.get("AWS_ACCESS_KEY_ID"):
        options["aws_access_key_id"] = key_id
    if secret := os.environ.get("AWS_SECRET_ACCESS_KEY"):
        options["aws_secret_access_key"] = secret
    if session_token := os.environ.get("AWS_SESSION_TOKEN"):
        options["aws_session_token"] = session_token

    options["client_kwargs"] = {
        "region_name": os.environ.get("AWS_DEFAULT_REGION", "eu-west-2")
    }

    return options


def prepare_pipeline(config) -> dict[str, Any]:
    validate_temporal_split(config)
    ensure_dirs(config)
    return get_storage_options()
