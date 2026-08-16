from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from entropy_model.config import EntropyModelConfig


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 5e-5
    betas: tuple[float, float] = (0.9, 0.97)
    weight_decay: float = 0.05
    grad_clip_norm: float = 1.0
    warmup_steps: int = 2000

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.grad_clip_norm <= 0:
            raise ValueError("Invalid optimizer scalar configuration")
        if len(self.betas) != 2 or not all(0 <= beta < 1 for beta in self.betas):
            raise ValueError(f"Invalid AdamW betas: {self.betas}")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")


@dataclass(frozen=True)
class TrainLoopConfig:
    batch_size: int = 4
    gradient_accumulation: int = 8
    max_steps: int = 60000
    num_workers: int = 4
    seed: int = 2026
    log_interval: int = 20
    checkpoint_interval: int = 5000
    bf16: bool = True

    def __post_init__(self) -> None:
        integer_values = (
            self.batch_size,
            self.gradient_accumulation,
            self.max_steps,
            self.log_interval,
            self.checkpoint_interval,
        )
        if any(value <= 0 for value in integer_values) or self.num_workers < 0:
            raise ValueError(f"Invalid train loop configuration: {self}")


@dataclass(frozen=True)
class ARTrainingConfig:
    model: EntropyModelConfig
    optimizer: OptimizerConfig
    train: TrainLoopConfig


T = TypeVar("T")


def _strict_construct(cls: type[T], values: dict[str, Any]) -> T:
    allowed = {field.name for field in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


def load_ar_training_config(path: str | Path) -> ARTrainingConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or set(raw) != {"model", "optimizer", "train"}:
        raise ValueError("AR config must contain exactly model, optimizer, and train mappings")
    if not all(isinstance(raw[key], dict) for key in raw):
        raise TypeError("AR config sections must be mappings")

    model_values = dict(raw["model"])
    if "levels" in model_values:
        model_values["levels"] = tuple(model_values["levels"])
    optimizer_values = dict(raw["optimizer"])
    if "betas" in optimizer_values:
        optimizer_values["betas"] = tuple(optimizer_values["betas"])
    return ARTrainingConfig(
        model=_strict_construct(EntropyModelConfig, model_values),
        optimizer=_strict_construct(OptimizerConfig, optimizer_values),
        train=_strict_construct(TrainLoopConfig, dict(raw["train"])),
    )
