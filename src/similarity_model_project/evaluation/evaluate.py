"""Evaluation logic for embedding-based product similarity."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

import polars as pl
from tqdm import tqdm

from similarity_model_project.evaluation.metrics import precision_at_k
from similarity_model_project.evaluation.ranking import (
    cosine_similarity,
    rank_by_cosine_similarity,
)
from similarity_model_project.preprocess.preprocess_product2vec import (
    Prod2VecDatasetResult,
    build_cooccurrence_ground_truth,
)
from similarity_model_project.similarity.base import BaseModel
from similarity_model_project.utils.model_io import upload_local_file_to_s3

TOP_CO_OCCURRENCE_LIMIT = 50
log = logging.getLogger(__name__)


class EvaluationConfigLike(Protocol):
    """Configuration attributes required by the evaluation pipeline."""

    pipeline_mode: str
    model_label: str
    local_results_dir: Path
    s3_bucket: str
    s3_results_prefix: str
    test_start_date: str
    test_end_date: str
    category_filter_enabled: bool
    retrieval_pool_size: int
    top_n_similar: int
    n_eval_products: int | None
    sample_products: list[str] | None


def build_readable_similarity_df(
    results_df: pl.DataFrame,
    mapping_df: pl.DataFrame,
) -> pl.DataFrame:
    """Convert nested evaluation output into a flat human-readable table."""
    category_lookup = dict(
        zip(mapping_df["ITEM_ID"].to_list(), mapping_df["CATEGORY"].to_list())
    )
    name_lookup = dict(
        zip(mapping_df["ITEM_ID"].to_list(), mapping_df["PRODUCT_NAME"].to_list())
    )

    readable_rows: list[dict[str, Any]] = []

    for row in results_df.iter_rows(named=True):
        focal_id = row["FOCAL_PRODUCT"]
        focal_category = category_lookup.get(focal_id)
        focal_name = name_lookup.get(focal_id)

        for related_id, score in zip(
            row["SUBSTITUTE_IDS"],
            row["SUBSTITUTE_SCORES"],
        ):
            readable_rows.append(
                {
                    "FOCAL_PRODUCT_ID": focal_id,
                    "FOCAL_PRODUCT_NAME": focal_name,
                    "FOCAL_CATEGORY": focal_category,
                    "RELATION_TYPE": "similar_product",
                    "RELATED_PRODUCT_ID": related_id,
                    "RELATED_PRODUCT_NAME": name_lookup.get(related_id),
                    "RELATED_CATEGORY": category_lookup.get(related_id),
                    "SCORE": score,
                }
            )

    return pl.DataFrame(readable_rows)


def format_products_for_log(
    product_ids: list[str],
    name_lookup: dict[str, str | None],
    category_lookup: dict[str, str | None],
) -> list[str]:
    """Format product IDs as 'PRODUCT_NAME | CATEGORY' for readable logging."""
    formatted: list[str] = []

    for pid in product_ids:
        product_name = name_lookup.get(pid)
        category = category_lookup.get(pid)

        if product_name is None:
            product_name = pid
        if category is None:
            category = "UNKNOWN_CATEGORY"

        formatted.append(f"{product_name} | {category}")

    return formatted


def build_worked_example(
    model: BaseModel,
    test_df: pl.DataFrame,
    mapping_df: pl.DataFrame,
    focal_product_id: str,
    category_filter_enabled: bool = False,
) -> dict[str, Any]:
    """Build a real worked evaluation example for one focal product."""
    embeddings = model.get_all_embeddings()
    if focal_product_id not in embeddings:
        raise ValueError(f"Focal product '{focal_product_id}' not found.")

    ground_truth = build_cooccurrence_ground_truth(
        test_df,
        top_n=TOP_CO_OCCURRENCE_LIMIT,
    )
    relevant_ids = sorted(list(ground_truth.get(focal_product_id, set())))

    name_lookup = dict(
        zip(mapping_df["ITEM_ID"].to_list(), mapping_df["PRODUCT_NAME"].to_list())
    )
    category_lookup = dict(
        zip(mapping_df["ITEM_ID"].to_list(), mapping_df["CATEGORY"].to_list())
    )

    focal_vector = embeddings[focal_product_id]
    focal_category = category_lookup.get(focal_product_id)

    scored: list[tuple[str, float]] = []
    for item_id, vec in embeddings.items():
        if item_id == focal_product_id:
            continue

        if category_filter_enabled and category_lookup.get(item_id) != focal_category:
            continue

        score = cosine_similarity(focal_vector, vec)
        scored.append((item_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    top_3 = scored[:3]
    top_5 = scored[:5]
    top_10 = scored[:10]

    top_3_ids = [pid for pid, _ in top_3]
    top_5_ids = [pid for pid, _ in top_5]
    top_10_ids = [pid for pid, _ in top_10]

    relevant_set = set(relevant_ids)

    intersection_3_ids = sorted(list(set(top_3_ids).intersection(relevant_set)))
    intersection_5_ids = sorted(list(set(top_5_ids).intersection(relevant_set)))
    intersection_10_ids = sorted(list(set(top_10_ids).intersection(relevant_set)))

    relevant_rows = [
        {
            "ITEM_ID": pid,
            "PRODUCT_NAME": name_lookup.get(pid),
            "CATEGORY": category_lookup.get(pid),
        }
        for pid in relevant_ids
        if name_lookup.get(pid) is not None and category_lookup.get(pid) is not None
    ]

    return {
        "FOCAL_PRODUCT_ID": focal_product_id,
        "FOCAL_PRODUCT_NAME": name_lookup.get(focal_product_id),
        "FOCAL_CATEGORY": focal_category,
        "FOCAL_EMBEDDING_FIRST_10_DIMS": [float(x) for x in focal_vector[:10].tolist()],
        "CATEGORY_FILTER_ENABLED": category_filter_enabled,
        "TOP_3_RESULTS": [
            {
                "ITEM_ID": pid,
                "PRODUCT_NAME": name_lookup.get(pid),
                "CATEGORY": category_lookup.get(pid),
                "SCORE": float(score),
                "RELATION_TYPE": "retrieved similar product",
                "IS_RELEVANT": pid in relevant_set,
            }
            for pid, score in top_3
        ],
        "TOP_5_RESULTS": [
            {
                "ITEM_ID": pid,
                "PRODUCT_NAME": name_lookup.get(pid),
                "CATEGORY": category_lookup.get(pid),
                "SCORE": float(score),
                "RELATION_TYPE": "retrieved similar product",
                "IS_RELEVANT": pid in relevant_set,
            }
            for pid, score in top_5
        ],
        "TOP_10_RESULTS": [
            {
                "ITEM_ID": pid,
                "PRODUCT_NAME": name_lookup.get(pid),
                "CATEGORY": category_lookup.get(pid),
                "SCORE": float(score),
                "RELATION_TYPE": "retrieved similar product",
                "IS_RELEVANT": pid in relevant_set,
            }
            for pid, score in top_10
        ],
        "RELEVANT_SET": relevant_rows,
        "RELEVANT_SET_SIZE": len(relevant_ids),
        "INTERSECTION_AT_3_IDS": intersection_3_ids,
        "INTERSECTION_AT_5_IDS": intersection_5_ids,
        "INTERSECTION_AT_10_IDS": intersection_10_ids,
        "INTERSECTION_AT_3_FORMATTED": format_products_for_log(
            intersection_3_ids,
            name_lookup,
            category_lookup,
        ),
        "INTERSECTION_AT_5_FORMATTED": format_products_for_log(
            intersection_5_ids,
            name_lookup,
            category_lookup,
        ),
        "INTERSECTION_AT_10_FORMATTED": format_products_for_log(
            intersection_10_ids,
            name_lookup,
            category_lookup,
        ),
        "PRECISION_AT_3": float(precision_at_k(top_3_ids, relevant_set, 3)),
        "PRECISION_AT_5": float(precision_at_k(top_5_ids, relevant_set, 5)),
        "PRECISION_AT_10": float(precision_at_k(top_10_ids, relevant_set, 10)),
    }


def log_worked_example(example: dict[str, Any]) -> None:
    """Log worked example clearly for dissertation use."""
    log.info("=" * 70)
    log.info("WORKED EXAMPLE")
    log.info("=" * 70)

    log.info(
        "Focal product: %s | %s | %s",
        example["FOCAL_PRODUCT_ID"],
        example["FOCAL_PRODUCT_NAME"],
        example["FOCAL_CATEGORY"],
    )
    log.info(
        "Worked example category restriction enabled: %s",
        example["CATEGORY_FILTER_ENABLED"],
    )

    log.info(
        "Focal embedding (first 10 dims): %s",
        [round(x, 4) for x in example["FOCAL_EMBEDDING_FIRST_10_DIMS"]],
    )

    log.info("Top-3 retrieved similar products:")
    for row in example["TOP_3_RESULTS"]:
        log.info(
            "  %s | %s | %s | score=%.4f | relevant=%s",
            row["ITEM_ID"],
            row["PRODUCT_NAME"],
            row["CATEGORY"],
            row["SCORE"],
            row["IS_RELEVANT"],
        )

    log.info("Top-5 retrieved similar products:")
    for row in example["TOP_5_RESULTS"]:
        log.info(
            "  %s | %s | %s | score=%.4f | relevant=%s",
            row["ITEM_ID"],
            row["PRODUCT_NAME"],
            row["CATEGORY"],
            row["SCORE"],
            row["IS_RELEVANT"],
        )

    log.info("Top-10 retrieved similar products:")
    for row in example["TOP_10_RESULTS"]:
        log.info(
            "  %s | %s | %s | score=%.4f | relevant=%s",
            row["ITEM_ID"],
            row["PRODUCT_NAME"],
            row["CATEGORY"],
            row["SCORE"],
            row["IS_RELEVANT"],
        )

    log.info(
        "Relevant set size (top-50 co-occurrence ground truth): %s",
        example["RELEVANT_SET_SIZE"],
    )
    log.info("Relevant set (top 20 shown, from top-50 co-occurrence ground truth):")
    for row in example["RELEVANT_SET"][:20]:
        log.info(
            "  %s | %s | %s",
            row["ITEM_ID"],
            row["PRODUCT_NAME"],
            row["CATEGORY"],
        )

    log.info("Intersection@3:")
    for row in example["INTERSECTION_AT_3_FORMATTED"]:
        log.info("  %s", row)

    log.info("Intersection@5:")
    for row in example["INTERSECTION_AT_5_FORMATTED"]:
        log.info("  %s", row)

    log.info("Intersection@10:")
    for row in example["INTERSECTION_AT_10_FORMATTED"]:
        log.info("  %s", row)

    log.info("Precision@3 = %.4f", example["PRECISION_AT_3"])
    log.info("Precision@5 = %.4f", example["PRECISION_AT_5"])
    log.info("Precision@10 = %.4f", example["PRECISION_AT_10"])


def save_worked_example(
    worked_example: dict[str, Any],
    config: EvaluationConfigLike,
    storage_options: dict[str, Any],
) -> None:
    """Save the worked example locally and upload it to S3."""
    model_label = getattr(config, "model_label", config.pipeline_mode)

    local_worked_example_path = (
        config.local_results_dir
        / f"worked_example_{model_label}_{config.test_start_date}_{config.test_end_date}.json"
    )

    with local_worked_example_path.open("w", encoding="utf-8") as file_obj:
        json.dump(worked_example, file_obj, indent=2, default=str)

    log.info("Worked example saved locally: %s", local_worked_example_path)

    s3_worked_example_path = (
        f"s3://{config.s3_bucket}/{config.s3_results_prefix}/"
        f"worked_example_{model_label}_{config.test_start_date}_{config.test_end_date}.json"
    )

    upload_local_file_to_s3(
        local_path=local_worked_example_path,
        s3_path=s3_worked_example_path,
        storage_options=storage_options,
    )

    log.info("Worked example saved to S3: %s", s3_worked_example_path)


def evaluate_embedding_model(
    model: BaseModel,
    test_result: Prod2VecDatasetResult,
    mapping_df: pl.DataFrame | None,
    config: EvaluationConfigLike,
    storage_options: dict[str, Any],
) -> pl.DataFrame:
    """Evaluate a similarity model using embeddings and cosine similarity."""
    model_label = getattr(config, "model_label", config.pipeline_mode)

    embeddings = model.get_all_embeddings()
    if not embeddings:
        raise ValueError("Model returned no embeddings for evaluation.")

    log.info("=" * 70)
    log.info("PHASE 4–5 — EMBEDDING-BASED EVALUATION")
    log.info("=" * 70)
    log.info("Category filtering enabled : %s", config.category_filter_enabled)
    log.info("Retrieval pool size        : %s", config.retrieval_pool_size)
    log.info("Ground-truth top-N filter  : %s", TOP_CO_OCCURRENCE_LIMIT)

    vocab = list(embeddings.keys())
    test_ground_truth = build_cooccurrence_ground_truth(
        test_result.dataset,
        top_n=TOP_CO_OCCURRENCE_LIMIT,
    )

    if config.sample_products:
        sample_products = list(config.sample_products)
    elif config.n_eval_products is None:
        sample_products = vocab
    else:
        sample_products = vocab[: config.n_eval_products]

    log.info(
        "Number of focal products for evaluation: %s",
        f"{len(sample_products):,}",
    )
    log.info(
        "Products with held-out relevance sets: %s",
        f"{len(test_ground_truth):,}",
    )

    category_lookup: dict[str, str] = {}
    name_lookup: dict[str, str] = {}
    if mapping_df is not None:
        category_lookup = dict(
            zip(
                mapping_df["ITEM_ID"].to_list(),
                mapping_df["CATEGORY"].to_list(),
            )
        )
        name_lookup = dict(
            zip(
                mapping_df["ITEM_ID"].to_list(),
                mapping_df["PRODUCT_NAME"].to_list(),
            )
        )

    rows: list[dict[str, Any]] = []

    log.info("Starting evaluation loop with progress bar...")

    for product in tqdm(
        sample_products,
        total=len(sample_products),
        desc="Evaluating products",
        unit="product",
    ):
        if product not in embeddings:
            log.warning("Product '%s' not in embeddings; skipping.", product)
            continue

        raw_ranked = rank_by_cosine_similarity(
            focal_item_id=product,
            embeddings=embeddings,
            topn=config.retrieval_pool_size,
        )

        focal_category = category_lookup.get(product)

        if (
            config.category_filter_enabled
            and mapping_df is not None
            and focal_category is not None
        ):
            substitutes = [
                (pid, score)
                for pid, score in raw_ranked
                if category_lookup.get(pid) == focal_category
            ][: config.top_n_similar]
        else:
            substitutes = raw_ranked[: config.top_n_similar]

        sub_ids = [item_id for item_id, _ in substitutes]
        sub_scores = [score for _, score in substitutes]

        separator = "─" * 60
        focal_name = name_lookup.get(product, product)
        focal_cat = category_lookup.get(product, "?")
        log.info("\n%s", separator)
        log.info("Product: %s | %s | %s", product, focal_name, focal_cat)
        log.info("%s", separator)
        log.info("Top similar products:")
        for pid, score in substitutes:
            pname = name_lookup.get(pid, pid)
            pcat = category_lookup.get(pid, "?")
            log.info("  %-25s %-40s %-25s %.4f", pid, pname, pcat, score)

        test_relevant_ids = test_ground_truth.get(product, set())

        test_precision_at_3_value = (
            precision_at_k(sub_ids, test_relevant_ids, 3) if test_relevant_ids else None
        )
        test_precision_at_5_value = (
            precision_at_k(sub_ids, test_relevant_ids, 5) if test_relevant_ids else None
        )
        test_precision_at_10_value = (
            precision_at_k(sub_ids, test_relevant_ids, 10)
            if test_relevant_ids
            else None
        )

        category_precision_at_3_value = None
        category_precision_at_5_value = None
        category_precision_at_10_value = None

        if mapping_df is not None and focal_category is not None:
            category_relevant_ids = {
                item_id
                for item_id, category in category_lookup.items()
                if category == focal_category and item_id != product
            }

            if category_relevant_ids:
                category_precision_at_3_value = precision_at_k(
                    sub_ids, category_relevant_ids, 3
                )
                category_precision_at_5_value = precision_at_k(
                    sub_ids, category_relevant_ids, 5
                )
                category_precision_at_10_value = precision_at_k(
                    sub_ids, category_relevant_ids, 10
                )

        log.info(
            (
                "%s | test_rel=%s | category=%s | "
                "TEST P@3=%.2f | P@5=%.2f | P@10=%.2f"
            ),
            product,
            len(test_relevant_ids),
            focal_category,
            (
                test_precision_at_3_value
                if test_precision_at_3_value is not None
                else -1.0
            ),
            (
                test_precision_at_5_value
                if test_precision_at_5_value is not None
                else -1.0
            ),
            (
                test_precision_at_10_value
                if test_precision_at_10_value is not None
                else -1.0
            ),
        )

        rows.append(
            {
                "FOCAL_PRODUCT": product,
                "SUBSTITUTE_IDS": sub_ids,
                "SUBSTITUTE_SCORES": sub_scores,
                "TEST_PRECISION_AT_3": test_precision_at_3_value,
                "TEST_PRECISION_AT_5": test_precision_at_5_value,
                "TEST_PRECISION_AT_10": test_precision_at_10_value,
                "CATEGORY_PRECISION_AT_3": category_precision_at_3_value,
                "CATEGORY_PRECISION_AT_5": category_precision_at_5_value,
                "CATEGORY_PRECISION_AT_10": category_precision_at_10_value,
            }
        )

    results_df = pl.DataFrame(
        rows,
        infer_schema_length=None,
        schema_overrides={
            "TEST_PRECISION_AT_3": pl.Float64,
            "TEST_PRECISION_AT_5": pl.Float64,
            "TEST_PRECISION_AT_10": pl.Float64,
            "CATEGORY_PRECISION_AT_3": pl.Float64,
            "CATEGORY_PRECISION_AT_5": pl.Float64,
            "CATEGORY_PRECISION_AT_10": pl.Float64,
        },
    )

    if results_df.height > 0:
        for metric in [
            "TEST_PRECISION_AT_3",
            "TEST_PRECISION_AT_5",
            "TEST_PRECISION_AT_10",
            "CATEGORY_PRECISION_AT_3",
            "CATEGORY_PRECISION_AT_5",
            "CATEGORY_PRECISION_AT_10",
        ]:
            if metric in results_df.columns:
                valid_scores = results_df[metric].drop_nulls()
                if len(valid_scores) > 0:
                    log.info("Mean %s: %.3f", metric, float(valid_scores.mean()))

    local_eval_path = (
        config.local_results_dir
        / f"evaluation_{model_label}_{config.test_start_date}_{config.test_end_date}.pq"
    )
    results_df.write_parquet(local_eval_path)
    log.info("Evaluation results saved locally: %s", local_eval_path)

    s3_eval_path = (
        f"s3://{config.s3_bucket}/{config.s3_results_prefix}/"
        f"evaluation_{model_label}_{config.test_start_date}_{config.test_end_date}.pq"
    )

    try:
        upload_local_file_to_s3(
            local_path=local_eval_path,
            s3_path=s3_eval_path,
            storage_options=storage_options,
        )
        log.info("Evaluation results saved to S3: %s", s3_eval_path)
    except Exception as err:
        log.warning(
            "Could not save evaluation results to S3. "
            "Local copy remains at %s. Error: %s",
            local_eval_path,
            err,
        )

    if mapping_df is not None and results_df.height > 0:
        readable_df = build_readable_similarity_df(results_df, mapping_df)

        local_readable_parquet_path = (
            config.local_results_dir
            / f"evaluation_readable_{model_label}_{config.test_start_date}_{config.test_end_date}.pq"
        )
        local_readable_csv_path = (
            config.local_results_dir
            / f"evaluation_readable_{model_label}_{config.test_start_date}_{config.test_end_date}.csv"
        )

        readable_df.write_parquet(local_readable_parquet_path)
        readable_df.write_csv(local_readable_csv_path)

        log.info(
            "Readable evaluation results saved locally: %s",
            local_readable_parquet_path,
        )
        log.info(
            "Readable evaluation CSV saved locally: %s",
            local_readable_csv_path,
        )

        s3_readable_parquet_path = (
            f"s3://{config.s3_bucket}/{config.s3_results_prefix}/"
            f"evaluation_readable_{model_label}_{config.test_start_date}_{config.test_end_date}.pq"
        )
        s3_readable_csv_path = (
            f"s3://{config.s3_bucket}/{config.s3_results_prefix}/"
            f"evaluation_readable_{model_label}_{config.test_start_date}_{config.test_end_date}.csv"
        )

        try:
            upload_local_file_to_s3(
                local_path=local_readable_parquet_path,
                s3_path=s3_readable_parquet_path,
                storage_options=storage_options,
            )
            upload_local_file_to_s3(
                local_path=local_readable_csv_path,
                s3_path=s3_readable_csv_path,
                storage_options=storage_options,
            )
            log.info(
                "Readable evaluation results saved to S3: %s",
                s3_readable_parquet_path,
            )
            log.info(
                "Readable evaluation CSV saved to S3: %s",
                s3_readable_csv_path,
            )
        except Exception as err:
            log.warning(
                "Could not save readable evaluation outputs to S3. "
                "Local CSV/parquet copies were still saved. Error: %s",
                err,
            )

    if mapping_df is not None and len(sample_products) > 0:
        try:
            valid_result_rows = results_df.drop_nulls(["TEST_PRECISION_AT_5"])

            if valid_result_rows.height == 0:
                raise ValueError(
                    "No valid evaluation rows available for worked example."
                )

            best_row = valid_result_rows.sort(
                by=["TEST_PRECISION_AT_5", "TEST_PRECISION_AT_10"],
                descending=True,
            ).row(0, named=True)

            example_product = best_row["FOCAL_PRODUCT"]

            name_lookup = dict(
                zip(
                    mapping_df["ITEM_ID"].to_list(),
                    mapping_df["PRODUCT_NAME"].to_list(),
                )
            )
            category_lookup = dict(
                zip(
                    mapping_df["ITEM_ID"].to_list(),
                    mapping_df["CATEGORY"].to_list(),
                )
            )

            log.info(
                "Selected focal product for worked example: %s | %s | %s",
                example_product,
                name_lookup.get(example_product),
                category_lookup.get(example_product),
            )

            worked_example = build_worked_example(
                model=model,
                test_df=test_result.dataset,
                mapping_df=mapping_df,
                focal_product_id=example_product,
                category_filter_enabled=config.category_filter_enabled,
            )

            log_worked_example(worked_example)
            save_worked_example(worked_example, config, storage_options)

        except Exception as err:
            log.warning("Worked example generation failed: %s", err)

    return results_df
