from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass(frozen=True)
class WanVQOptimizerConfig:
    projection_learning_rate: float
    codebook_learning_rate: float
    betas: tuple[float, float]
    projection_weight_decay: float
    grad_clip_norm: float
    warmup_steps: int

    def __post_init__(self) -> None:
        positive = (
            self.projection_learning_rate,
            self.codebook_learning_rate,
            self.grad_clip_norm,
            self.warmup_steps,
        )
        if any(value <= 0 for value in positive):
            raise ValueError(f"Invalid optimizer configuration: {self}")
        if self.projection_weight_decay < 0:
            raise ValueError("projection_weight_decay must be non-negative")
        if len(self.betas) != 2 or not all(0 <= beta < 1 for beta in self.betas):
            raise ValueError(f"Invalid AdamW betas: {self.betas}")


@dataclass(frozen=True)
class WanVQLossConfig:
    latent_weight: float
    pixel_weight: float
    lpips_weight: float
    codebook_weight: float
    commitment_weight: float

    def __post_init__(self) -> None:
        if any(value <= 0 for value in vars(self).values()):
            raise ValueError(f"All Stage 1 loss weights must be positive: {self}")


@dataclass(frozen=True)
class WanVQTrainConfig:
    batch_size: int
    gradient_accumulation: int
    max_steps: int
    num_workers: int
    seed: int
    log_interval: int
    checkpoint_interval: int
    kmeans_iterations: int
    minimum_active_codes: int

    def __post_init__(self) -> None:
        if any(value <= 0 for value in vars(self).values()):
            raise ValueError(f"All Stage 1 train values must be positive: {self}")
        if self.minimum_active_codes > 4096:
            raise ValueError("minimum_active_codes cannot exceed 4096")


@dataclass(frozen=True)
class WanVQTrainingConfig:
    optimizer: WanVQOptimizerConfig
    loss: WanVQLossConfig
    train: WanVQTrainConfig

    def __post_init__(self) -> None:
        if self.optimizer.warmup_steps >= self.train.max_steps:
            raise ValueError("warmup_steps must be smaller than max_steps")


T = TypeVar("T")


def _strict_construct(cls: type[T], values: dict[str, Any]) -> T:
    required = {field.name for field in fields(cls)}
    if set(values) != required:
        raise ValueError(
            f"{cls.__name__} requires {sorted(required)}, got {sorted(values)}"
        )
    return cls(**values)


def load_wan_vq_training_config(path: str | Path) -> WanVQTrainingConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    sections = {"optimizer", "loss", "train"}
    if not isinstance(raw, dict) or set(raw) != sections:
        raise ValueError(f"Config must contain exactly {sorted(sections)}")
    optimizer = dict(raw["optimizer"])
    optimizer["betas"] = tuple(optimizer["betas"])
    return WanVQTrainingConfig(
        optimizer=_strict_construct(WanVQOptimizerConfig, optimizer),
        loss=_strict_construct(WanVQLossConfig, dict(raw["loss"])),
        train=_strict_construct(WanVQTrainConfig, dict(raw["train"])),
    )
