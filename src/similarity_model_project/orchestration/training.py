# ruff: noqa: D103, ANN001
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from similarity_model_project.preprocess.preprocess_product2vec import (
    Prod2VecDatasetResult,
    baskets_for_prod2vec,
    build_cooccurrence_counts,
    validate_aggregated_basket_df,
)
from similarity_model_project.preprocess.preprocess_transformer import (
    build_transformer_products,
    preprocess_transformer_training_pairs,
)
from similarity_model_project.similarity.product2vec import Product2Vec
from similarity_model_project.similarity.sentence_transformer import (
    SentenceTransformerModel,
)
from similarity_model_project.utils.model_io import save_model_locally

log = logging.getLogger(__name__)


def get_torch_device() -> str:
    """Return CUDA if available, otherwise CPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device          : %s", device)

    if device == "cuda":
        log.info("CUDA device name      : %s", torch.cuda.get_device_name(0))

    return device


def phase_3_train_product2vec(
    result: Prod2VecDatasetResult,
    config,
) -> tuple[Product2Vec, Path]:
    log.info("=" * 70)
    log.info("PHASE 3 — PRODUCT2VEC MODEL")
    log.info("Vector size          : %s", config.vector_size)
    log.info("Window               : %s", config.window)
    log.info("Epochs               : %s", config.epochs)
    log.info("Min count            : %s", config.min_count)
    log.info("Workers              : %s", config.workers)
    log.info("Seed                 : %s", config.seed)
    log.info("=" * 70)

    validate_aggregated_basket_df(result.dataset)

    sentences = baskets_for_prod2vec(result.dataset)
    log.info("Training baskets     : %s", f"{len(sentences):,}")

    model = Product2Vec(
        vector_size=config.vector_size,
        window=config.window,
        epochs=config.epochs,
        min_count=config.min_count,
        workers=config.workers,
        seed=config.seed,
    )

    model.fit(sentences)

    if model.model_ is None:
        raise ValueError("Product2Vec model training failed; internal model is None.")

    log.info("Training complete. Vocabulary size: %s", f"{len(model.model_.wv):,}")

    model_path = config.local_results_dir / (
        f"prod2vec_"
        f"{config.train_start_date}_{config.train_end_date}_"
        f"{result.dataset_hash}.pkl"
    )

    save_model_locally(model=model, local_path=model_path)

    return model, model_path


def phase_3_build_sentence_transformer(
    mapping_df: pl.DataFrame,
    config,
) -> SentenceTransformerModel:
    log.info("=" * 70)
    log.info("PHASE 3 — SENTENCE TRANSFORMER BASELINE")
    log.info("Model name           : %s", config.sentence_transformer_model_name)
    log.info("Text mode            : %s", config.transformer_text_mode)
    log.info("=" * 70)

    device = get_torch_device()

    product_pairs = build_transformer_products(
        mapping_df=mapping_df,
        text_mode=config.transformer_text_mode,
    )

    if config.max_transformer_products is not None:
        product_pairs = product_pairs[: config.max_transformer_products]

    if len(product_pairs) == 0:
        raise ValueError("No products available for sentence-transformer encoding.")

    log.info("Products to encode   : %s", f"{len(product_pairs):,}")

    model = SentenceTransformerModel(
        model_name=config.sentence_transformer_model_name,
    )

    model.fit(product_pairs)

    log.info(
        "Encoding complete. Vocabulary size: %s",
        f"{len(model.get_all_embeddings()):,}",
    )

    return model


def freeze_transformer_layers(
    model: SentenceTransformer,
    trainable_layers: int | None,
) -> None:
    """Freeze transformer layers except the final N layers."""
    if trainable_layers is None:
        log.info("Full fine-tuning enabled. All transformer layers are trainable.")
        return

    transformer = model[0].auto_model

    for param in transformer.parameters():
        param.requires_grad = False

    encoder_layers = transformer.encoder.layer
    total_layers = len(encoder_layers)

    if trainable_layers < 0:
        raise ValueError("trainable_layers must be None, 0, or a positive integer.")

    if trainable_layers > total_layers:
        raise ValueError(
            f"Requested {trainable_layers} trainable layers, "
            f"but model only has {total_layers} encoder layers."
        )

    for layer in encoder_layers[-trainable_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    total_params = sum(param.numel() for param in model.parameters())

    log.info(
        "Layer freezing enabled: %s/%s final transformer layers trainable.",
        trainable_layers,
        total_layers,
    )
    log.info(
        "Trainable parameters: %s / %s",
        f"{trainable_params:,}",
        f"{total_params:,}",
    )


def phase_3_train_fine_tuned_sentence_transformer(
    train_result: Prod2VecDatasetResult,
    mapping_df: pl.DataFrame,
    config,
) -> SentenceTransformerModel:
    log.info("=" * 70)
    log.info("PHASE 3 — FINE-TUNED SENTENCE TRANSFORMER")
    log.info("Base model           : %s", config.sentence_transformer_model_name)
    log.info("Text mode            : %s", config.transformer_text_mode)
    log.info("Trainable layers     : %s", config.transformer_trainable_layers)
    log.info("Learning rate        : %s", config.transformer_learning_rate)
    log.info("=" * 70)

    device = get_torch_device()

    cooccurrence_counts = build_cooccurrence_counts(train_result.dataset)

    cooccurrence_df = pl.DataFrame(
        [
            {
                "PRODUCT_A": focal_product,
                "PRODUCT_B": related_product,
                "CO_OCCURRENCE_COUNT": count,
            }
            for focal_product, related_counts in cooccurrence_counts.items()
            for related_product, count in sorted(
                related_counts.items(), key=lambda x: -x[1]
            )[: config.ground_truth_top_n]
        ]
    )

    training_data = preprocess_transformer_training_pairs(
        mapping_df=mapping_df,
        cooccurrence_df=cooccurrence_df,
        text_mode=config.transformer_text_mode,
        min_cooccurrence_count=config.transformer_min_cooccurrence_count,
        max_pairs=config.transformer_max_training_pairs,
    )

    examples = [
        InputExample(texts=[row["TEXT_A"], row["TEXT_B"]], label=1.0)
        for row in training_data.positive_pairs.iter_rows(named=True)
    ]

    if len(examples) == 0:
        raise ValueError("No sentence-transformer training pairs were created.")

    log.info("Training pairs       : %s", f"{len(examples):,}")

    base_model = SentenceTransformer(
        config.sentence_transformer_model_name,
        device=device,
    )

    base_model.to(device)

    freeze_transformer_layers(
        model=base_model,
        trainable_layers=config.transformer_trainable_layers,
    )

    train_dataloader = DataLoader(
        examples,
        shuffle=True,
        batch_size=config.transformer_batch_size,
    )

    train_loss = losses.CosineSimilarityLoss(base_model)

    base_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=config.transformer_epochs,
        warmup_steps=config.transformer_warmup_steps,
        optimizer_params={"lr": config.transformer_learning_rate},
        show_progress_bar=True,
        use_amp=device == "cuda",
    )

    learning_rate_label = (
        str(config.transformer_learning_rate).replace("-", "minus").replace(".", "p")
    )

    layer_label = (
        "all_layers"
        if config.transformer_trainable_layers is None
        else f"last_{config.transformer_trainable_layers}_layers"
    )

    local_model_dir = config.local_results_dir / (
        f"fine_tuned_sentence_transformer_"
        f"{layer_label}_"
        f"lr_{learning_rate_label}_"
        f"{config.train_start_date}_{config.train_end_date}_"
        f"{train_result.dataset_hash}"
    )

    base_model.save(str(local_model_dir))

    log.info("Saved fine-tuned model to: %s", local_model_dir)

    product_pairs = build_transformer_products(
        mapping_df=mapping_df,
        text_mode=config.transformer_text_mode,
    )

    if config.max_transformer_products is not None:
        product_pairs = product_pairs[: config.max_transformer_products]

    model = SentenceTransformerModel(
        model_name=str(local_model_dir),
    )

    model.fit(product_pairs)

    return model
