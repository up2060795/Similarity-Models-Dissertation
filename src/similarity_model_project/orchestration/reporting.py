# ruff: noqa: D103, ANN001
from __future__ import annotations

import logging

import polars as pl

log = logging.getLogger(__name__)


def log_pipeline_summary(
    config,
    raw_df: pl.DataFrame,
    result,
    model,
    eval_df: pl.DataFrame,
) -> None:
    if model.model_ is None:
        raise ValueError("Product2Vec model has not been trained.")

    log.info("\n%s", "═" * 70)
    log.info("PIPELINE SUMMARY")
    log.info("%s", "═" * 70)

    log.info(
        "Train date range         : %s → %s",
        config.train_start_date,
        config.train_end_date,
    )

    log.info(
        "Test date range          : %s → %s",
        config.test_start_date,
        config.test_end_date,
    )

    log.info("Raw rows loaded          : %s", f"{raw_df.height:,}")
    log.info("Processed baskets        : %s", f"{result.dataset.height:,}")
    log.info("Dataset hash             : %s", result.dataset_hash)
    log.info("Vocabulary size          : %s", f"{len(model.model_.wv):,}")
    log.info("Vector size              : %s", config.vector_size)
    log.info("Window                   : %s", config.window)
    log.info("Epochs                   : %s", config.epochs)
    log.info("Products evaluated       : %s", eval_df.height)

    for label in [
        "TEST_PRECISION_AT_3",
        "TEST_PRECISION_AT_5",
        "TEST_PRECISION_AT_10",
    ]:
        if label in eval_df.columns:
            valid = eval_df[label].drop_nulls()

            if len(valid) > 0:
                log.info("Mean %s : %.4f", label, float(valid.mean()))

    log.info("%s", "═" * 70)
