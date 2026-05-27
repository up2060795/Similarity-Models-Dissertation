"""Entry point for the similarity-model pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from similarity_model_project.orchestration.orchestrate import (
    run_evaluation_only_pipeline,
    run_fine_tuned_sentence_transformer_pipeline,
    run_full_pipeline,
    run_model_comparison_pipeline,
    run_sentence_transformer_pipeline,
    run_training_pipeline,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for a single pipeline run."""

    train_start_date: date = date(2025, 1, 1)
    train_end_date: date = date(2025, 12, 31)

    test_start_date: date = date(2026, 1, 1)
    test_end_date: date = date(2026, 1, 27)

    s3_bucket: str = "similarity-models-student-project-data"
    s3_basket_prefix: str = "basket_data"
    s3_mapping_key: str = "lookups/item_lookup.pq"
    s3_results_prefix: str = "results"

    local_data_dir: Path = Path("data")
    local_processed_dir: Path = Path("data/processed")
    local_results_dir: Path = Path("data/results")

    basket_id_col: str = "BASKET_ID"
    item_id_col: str = "ITEM_ID"
    category_col: str = "SUBFIN_DESC"

    min_basket_size: int = 3
    deduplicate_items: bool = False

    vector_size: int = 100
    window: int = 5
    epochs: int = 10
    min_count: int = 5
    workers: int = 1
    seed: int = 42

    top_n_similar: int = 10
    retrieval_pool_size: int = 200
    category_filter_enabled: bool = False

    use_cached_data: bool = (
        True  # Set to False to re-run data loading and preprocessing steps
    )

    n_eval_products: int | None = 12_044  # None means evaluate on all products
    sample_products: tuple[str, ...] = ()
    max_baskets_for_dev: int | None = 5_000_000  # None means no limit

    sentence_transformer_model_name: str = "all-MiniLM-L6-v2"
    transformer_text_mode: str = "name_and_category"
    max_transformer_products: int | None = 12_044

    # -----------------------------
    # Fine-tuned Sentence Transformer
    # -----------------------------
    transformer_min_cooccurrence_count: int = 5
    transformer_max_training_pairs: int | None = 50_000
    transformer_batch_size: int = 512
    transformer_epochs: int = 1
    transformer_warmup_steps: int = 100
    ground_truth_top_n: int = 20

    # Freezing layers
    transformer_trainable_layers: int | None = 1  # None means all 6 layers trainable
    transformer_learning_rate: float = 2e-6

    pipeline_mode: str = (
        "full"  # "train", "evaluate", "full", "fine_tuned/sentence_transformer, model_comparison"
    )
    model_dataset_hash: str | None = "d35fcb39aa09a153"
    model_label: str = (
        "product2vec"  # "product2vec", "sentence_transformer", "fine_tuned_sentence_transformer"
    )

    # comparison settings for evaluation-only pipeline
    anchor_product_id: str | None = "01059618|20160524"  # None = "02169934|20181214"
    comparison_top_k: int = 10


def main() -> None:
    """Run the configured pipeline mode."""
    config = PipelineConfig()

    if config.pipeline_mode == "train":
        run_training_pipeline(config)

    elif config.pipeline_mode == "evaluate":
        if config.model_dataset_hash is None:
            raise ValueError(
                "model_dataset_hash must be set when pipeline_mode='evaluate'."
            )
        run_evaluation_only_pipeline(
            config=config,
            dataset_hash=config.model_dataset_hash,
        )
    elif config.pipeline_mode == "full":
        run_full_pipeline(config)

    elif config.pipeline_mode == "sentence_transformer":
        run_sentence_transformer_pipeline(config)

    elif config.pipeline_mode == "fine_tuned_sentence_transformer":
        run_fine_tuned_sentence_transformer_pipeline(config)
    elif config.pipeline_mode == "model_comparison":
        run_model_comparison_pipeline(config)

    else:
        raise ValueError(
            f"Unknown pipeline_mode '{config.pipeline_mode}'. "
            "Expected one of: 'train', 'evaluate', 'full', "
            "'sentence_transformer', 'fine_tuned_sentence_transformer', "
            "'model_comparison'."
        )


if __name__ == "__main__":
    main()
