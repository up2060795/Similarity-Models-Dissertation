# ruff: noqa: D103, ANN001
from __future__ import annotations

import logging

import polars as pl

from similarity_model_project.evaluation.evaluate import evaluate_embedding_model
from similarity_model_project.evaluation.model_comparison import (
    build_anchor_product_comparison,
    build_dissertation_comparison_table,
)
from similarity_model_project.orchestration.data import (
    load_standardised_mapping,
    phase_1_load_train_test_data,
)
from similarity_model_project.orchestration.preprocessing import phase_2_preprocess
from similarity_model_project.orchestration.reporting import log_pipeline_summary
from similarity_model_project.orchestration.setup import prepare_pipeline
from similarity_model_project.orchestration.training import (
    phase_3_build_sentence_transformer,
    phase_3_train_fine_tuned_sentence_transformer,
    phase_3_train_product2vec,
)
from similarity_model_project.utils.model_io import (
    load_model_from_s3,
    save_embeddings,
    save_model_to_s3,
    save_run_metadata,
)

log = logging.getLogger(__name__)


def get_or_create_processed_result(
    raw_df,
    config,
    storage_options,
    dataset_label: str,
):
    """Load processed data from cache if available, otherwise preprocess normally."""
    date_start = (
        config.train_start_date if dataset_label == "train" else config.test_start_date
    )
    date_end = (
        config.train_end_date if dataset_label == "train" else config.test_end_date
    )

    limit_label = (
        "all"
        if config.max_baskets_for_dev is None
        else f"limit_{config.max_baskets_for_dev}"
    )

    cache_path = (
        config.local_processed_dir
        / f"processed_{dataset_label}_baskets_{date_start}_{date_end}_{limit_label}.pq"
    )

    config.local_processed_dir.mkdir(parents=True, exist_ok=True)

    use_cache = getattr(config, "use_cached_data", True)

    if use_cache and cache_path.exists():
        log.info("Loading cached processed %s dataset: %s", dataset_label, cache_path)

        from similarity_model_project.preprocess.preprocess_product2vec import (
            Prod2VecDatasetResult,
        )

        cached_df = pl.read_parquet(cache_path)

        dataset_hash = f"cached_{dataset_label}_{date_start}_{date_end}_{limit_label}"

        return Prod2VecDatasetResult(
            dataset=cached_df,
            dataset_hash=dataset_hash,
            output_path=cache_path,
        )

    result = phase_2_preprocess(
        raw_df=raw_df,
        config=config,
        storage_options=storage_options,
        dataset_label=dataset_label,
    )

    result.dataset.write_parquet(cache_path)
    log.info("Cached processed %s dataset: %s", dataset_label, cache_path)

    return result


def run_training_pipeline(config) -> tuple[str, str]:
    """Train Product2Vec and upload the model to S3."""
    storage_options = prepare_pipeline(config)

    log.info("Starting training-only Product2Vec pipeline")

    train_df, _ = phase_1_load_train_test_data(config, storage_options)

    train_result = get_or_create_processed_result(
        raw_df=train_df,
        config=config,
        storage_options=storage_options,
        dataset_label="train",
    )

    model, model_path = phase_3_train_product2vec(train_result, config)

    s3_model_path = save_model_to_s3(
        model_path=model_path,
        s3_bucket=config.s3_bucket,
        s3_results_prefix=config.s3_results_prefix,
        train_start_date=config.train_start_date,
        train_end_date=config.train_end_date,
        dataset_hash=train_result.dataset_hash,
        storage_options=storage_options,
    )

    log.info("Training-only pipeline complete.")
    log.info("Local model path: %s", model_path)
    log.info("S3 model path   : %s", s3_model_path)

    return train_result.dataset_hash, s3_model_path


def run_sentence_transformer_pipeline(config) -> None:
    """Build sentence-transformer embeddings and evaluate them."""
    storage_options = prepare_pipeline(config)

    log.info("Starting sentence-transformer baseline pipeline")

    _, test_df = phase_1_load_train_test_data(config, storage_options)

    test_result = get_or_create_processed_result(
        raw_df=test_df,
        config=config,
        storage_options=storage_options,
        dataset_label="test",
    )

    mapping_df = load_standardised_mapping(config, storage_options)

    model = phase_3_build_sentence_transformer(
        mapping_df=mapping_df,
        config=config,
    )

    eval_df = evaluate_embedding_model(
        model=model,
        test_result=test_result,
        mapping_df=mapping_df,
        config=config,
        storage_options=storage_options,
    )

    log.info("Sentence-transformer baseline pipeline complete.")
    log.info("Products evaluated: %s", eval_df.height)


def run_fine_tuned_sentence_transformer_pipeline(config) -> None:
    """Fine-tune sentence transformer, build embeddings, and evaluate."""
    storage_options = prepare_pipeline(config)

    log.info("Starting fine-tuned sentence-transformer pipeline")

    train_df, test_df = phase_1_load_train_test_data(config, storage_options)

    train_result = get_or_create_processed_result(
        raw_df=train_df,
        config=config,
        storage_options=storage_options,
        dataset_label="train",
    )

    test_result = get_or_create_processed_result(
        raw_df=test_df,
        config=config,
        storage_options=storage_options,
        dataset_label="test",
    )

    mapping_df = load_standardised_mapping(config, storage_options)

    model = phase_3_train_fine_tuned_sentence_transformer(
        train_result=train_result,
        mapping_df=mapping_df,
        config=config,
    )

    eval_df = evaluate_embedding_model(
        model=model,
        test_result=test_result,
        mapping_df=mapping_df,
        config=config,
        storage_options=storage_options,
    )

    log.info("Fine-tuned sentence-transformer pipeline complete.")
    log.info("Products evaluated: %s", eval_df.height)


def run_evaluation_only_pipeline(
    config,
    dataset_hash: str,
) -> None:
    """Load a trained model from S3 and evaluate it separately."""
    storage_options = prepare_pipeline(config)

    log.info("Starting evaluation-only Product2Vec pipeline")
    log.info("Using dataset hash: %s", dataset_hash)

    _, test_df = phase_1_load_train_test_data(config, storage_options)

    test_result = get_or_create_processed_result(
        raw_df=test_df,
        config=config,
        storage_options=storage_options,
        dataset_label="test",
    )

    model, local_model_path, s3_model_path = load_model_from_s3(
        s3_bucket=config.s3_bucket,
        s3_results_prefix=config.s3_results_prefix,
        train_start_date=config.train_start_date,
        train_end_date=config.train_end_date,
        dataset_hash=dataset_hash,
        local_model_dir=config.local_results_dir,
        storage_options=storage_options,
    )

    log.info("Loaded model from S3 path: %s", s3_model_path)
    log.info("Downloaded model to local path: %s", local_model_path)

    mapping_df = load_standardised_mapping(config, storage_options)

    eval_df = evaluate_embedding_model(
        model=model,
        test_result=test_result,
        mapping_df=mapping_df,
        config=config,
        storage_options=storage_options,
    )

    log.info("Evaluation-only pipeline complete.")
    log.info("Products evaluated: %s", eval_df.height)


def run_full_pipeline(config) -> None:
    """Train, save, then immediately evaluate."""
    storage_options = prepare_pipeline(config)

    log.info("Starting full Product2Vec pipeline")

    train_df, test_df = phase_1_load_train_test_data(config, storage_options)

    train_result = get_or_create_processed_result(
        raw_df=train_df,
        config=config,
        storage_options=storage_options,
        dataset_label="train",
    )

    test_result = get_or_create_processed_result(
        raw_df=test_df,
        config=config,
        storage_options=storage_options,
        dataset_label="test",
    )

    model, model_path = phase_3_train_product2vec(train_result, config)

    save_model_to_s3(
        model_path=model_path,
        s3_bucket=config.s3_bucket,
        s3_results_prefix=config.s3_results_prefix,
        train_start_date=config.train_start_date,
        train_end_date=config.train_end_date,
        dataset_hash=train_result.dataset_hash,
        storage_options=storage_options,
    )

    mapping_df = load_standardised_mapping(config, storage_options)

    eval_df = evaluate_embedding_model(
        model=model,
        test_result=test_result,
        mapping_df=mapping_df,
        config=config,
        storage_options=storage_options,
    )

    save_embeddings(
        model=model,
        mapping_df=mapping_df,
        config=config,
        storage_options=storage_options,
    )

    save_run_metadata(
        config=config,
        raw_df=train_df,
        result=train_result,
        model=model,
        eval_df=eval_df,
        model_path=model_path,
        storage_options=storage_options,
    )

    log_pipeline_summary(
        config=config,
        raw_df=train_df,
        result=train_result,
        model=model,
        eval_df=eval_df,
    )

    log.info("Full pipeline complete.")


def run_model_comparison_pipeline(config) -> None:
    """Train all three models and compare them on the same anchor product."""
    storage_options = prepare_pipeline(config)

    if config.anchor_product_id is None:
        raise ValueError("anchor_product_id must be set when running model comparison.")

    log.info("Starting three-model anchor comparison pipeline")

    train_df, test_df = phase_1_load_train_test_data(config, storage_options)

    train_result = get_or_create_processed_result(
        raw_df=train_df,
        config=config,
        storage_options=storage_options,
        dataset_label="train",
    )

    _ = get_or_create_processed_result(
        raw_df=test_df,
        config=config,
        storage_options=storage_options,
        dataset_label="test",
    )

    mapping_df = load_standardised_mapping(config, storage_options)

    product2vec_model, _ = phase_3_train_product2vec(
        result=train_result,
        config=config,
    )

    baseline_transformer_model = phase_3_build_sentence_transformer(
        mapping_df=mapping_df,
        config=config,
    )

    fine_tuned_transformer_model = phase_3_train_fine_tuned_sentence_transformer(
        train_result=train_result,
        mapping_df=mapping_df,
        config=config,
    )

    common_vocab = set(product2vec_model.get_all_embeddings().keys())
    common_vocab &= set(baseline_transformer_model.get_all_embeddings().keys())
    common_vocab &= set(fine_tuned_transformer_model.get_all_embeddings().keys())

    if not common_vocab:
        raise ValueError("No products are shared across all three model vocabularies.")

    anchor_item_id = config.anchor_product_id
    if anchor_item_id not in common_vocab:
        # Pick the highest-frequency product from the intersection using
        # Product2Vec's frequency-ordered vocab (index_to_key is freq-sorted).
        anchor_item_id = next(
            p for p in product2vec_model.model_.wv.index_to_key if p in common_vocab
        )
        log.warning(
            "Configured anchor '%s' not in all model vocabularies. "
            "Auto-selected '%s' (most frequent product in intersection).",
            config.anchor_product_id,
            anchor_item_id,
        )

    log.info("Anchor product for comparison: %s", anchor_item_id)

    comparison_df = build_anchor_product_comparison(
        anchor_item_id=anchor_item_id,
        mapping_df=mapping_df,
        product2vec_model=product2vec_model,
        baseline_transformer_model=baseline_transformer_model,
        fine_tuned_transformer_model=fine_tuned_transformer_model,
        k=config.comparison_top_k,
    )

    comparison_path = (
        config.local_results_dir
        / f"anchor_product_comparison_{anchor_item_id.replace('|', '_')}.csv"
    )

    comparison_df.write_csv(comparison_path)

    dissertation_df = build_dissertation_comparison_table(comparison_df)

    dissertation_path = (
        config.local_results_dir
        / f"anchor_product_comparison_dissertation_{anchor_item_id.replace('|', '_')}.csv"
    )

    dissertation_df.write_csv(dissertation_path)

    log.info("Anchor comparison saved to: %s", comparison_path)
    log.info("Dissertation comparison table saved to: %s", dissertation_path)

    log.info("Anchor comparison:")
    log.info("\n%s", comparison_df)

    log.info("Dissertation comparison table:")
    log.info("\n%s", dissertation_df)

    log.info("Three-model anchor comparison pipeline complete.")
