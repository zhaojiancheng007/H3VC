from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from entropy_model.config_categorical import CategoricalEntropyModelConfig
from training.config_stage2_cosine import (
    Stage2OptimizerConfig,
    Stage2TrainConfig,
    learning_rate_at_step,
)


@dataclass(frozen=True)
class ContextCorruptionConfig:
    sample_probability: float
    loss_weight: float
    knn_k: int
    min_retained_groups: int
    max_retained_groups: int


@dataclass(frozen=True)
class Stage2ContextCorruptionConfig:
    model: CategoricalEntropyModelConfig
    optimizer: Stage2OptimizerConfig
    train: Stage2TrainConfig
    context_corruption: ContextCorruptionConfig


T = TypeVar("T")


def _construct(cls: type[T], values: dict[str, Any]) -> T:
    expected = {field.name for field in fields(cls)}
    if set(values) != expected:
        raise ValueError(
            f"{cls.__name__} requires exactly {sorted(expected)}, "
            f"got {sorted(values)}"
        )
    return cls(**values)


def stage2_context_corruption_config_dict(
    config: Stage2ContextCorruptionConfig,
) -> dict[str, Any]:
    return asdict(config)


def load_stage2_context_corruption_config(
    path: str | Path,
) -> Stage2ContextCorruptionConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    required_sections = {"model", "optimizer", "train", "context_corruption"}
    if not isinstance(raw, dict) or set(raw) != required_sections:
        raise ValueError(f"Stage2 context-corruption config requires {required_sections}")
    optimizer = dict(raw["optimizer"])
    optimizer["betas"] = tuple(optimizer["betas"])
    config = Stage2ContextCorruptionConfig(
        model=_construct(CategoricalEntropyModelConfig, dict(raw["model"])),
        optimizer=_construct(Stage2OptimizerConfig, optimizer),
        train=_construct(Stage2TrainConfig, dict(raw["train"])),
        context_corruption=_construct(
            ContextCorruptionConfig, dict(raw["context_corruption"])
        ),
    )
    if min(
        config.optimizer.grad_clip_norm,
        config.train.batch_size,
        config.train.max_steps,
        config.train.num_workers,
        config.train.log_interval,
        config.train.checkpoint_interval,
        config.context_corruption.loss_weight,
        config.context_corruption.knn_k,
        config.context_corruption.min_retained_groups,
    ) <= 0:
        raise ValueError(f"Stage2 values must be positive: {config}")
    if (
        config.optimizer.weight_decay < 0
        or not config.optimizer.learning_rate > config.optimizer.eta_min >= 0
    ):
        raise ValueError(f"Invalid Stage2 optimizer values: {config.optimizer}")
    if config.train.max_steps <= 1:
        raise ValueError("Cosine LR requires max_steps > 1")
    if config.train.checkpoint_policy != "best_bpp":
        raise ValueError(
            f"Unknown Stage2 checkpoint policy: {config.train.checkpoint_policy}"
        )
    corruption = config.context_corruption
    corrupt_samples = config.train.batch_size * corruption.sample_probability
    if not corrupt_samples.is_integer() or int(corrupt_samples) != 1:
        raise ValueError(
            "Each local batch must contain exactly one corrupted sample, got "
            f"batch_size={config.train.batch_size}, "
            f"sample_probability={corruption.sample_probability}"
        )
    if not (
        1
        <= corruption.min_retained_groups
        <= corruption.max_retained_groups
        < 4
    ):
        raise ValueError("retained group range must be within [1,3]")
    if corruption.knn_k >= config.model.codebook_size:
        raise ValueError("knn_k must be smaller than codebook_size")
    return config


__all__ = [
    "Stage2ContextCorruptionConfig",
    "learning_rate_at_step",
    "load_stage2_context_corruption_config",
    "stage2_context_corruption_config_dict",
]
