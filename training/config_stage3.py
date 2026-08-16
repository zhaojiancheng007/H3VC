from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from entropy_model.config import EntropyModelConfig


@dataclass(frozen=True)
class Stage3OptimizerConfig:
    tokenizer_learning_rate: float
    ar_learning_rate: float
    betas: tuple[float, float]
    grad_clip_norm: float

    def __post_init__(self) -> None:
        if self.tokenizer_learning_rate <= 0 or self.ar_learning_rate <= 0:
            raise ValueError("Stage 3 learning rates must be positive")
        if len(self.betas) != 2 or not all(0 <= beta < 1 for beta in self.betas):
            raise ValueError(f"Invalid Adam betas: {self.betas}")
        if self.grad_clip_norm <= 0:
            raise ValueError(f"Invalid Stage 3 optimizer configuration: {self}")


@dataclass(frozen=True)
class Stage3LossConfig:
    mse_weight: float
    lpips_weight: float
    rate_weight: float
    surrogate_temperature: float
    lpips_net: str

    def __post_init__(self) -> None:
        if min(self.mse_weight, self.lpips_weight, self.rate_weight) <= 0:
            raise ValueError("MSE, LPIPS, and rate weights must all be positive")
        if self.surrogate_temperature <= 0:
            raise ValueError("surrogate_temperature must be positive")
        if self.lpips_net not in {"alex", "vgg"}:
            raise ValueError(f"lpips_net must be alex or vgg, got {self.lpips_net!r}")


@dataclass(frozen=True)
class Stage3TrainLoopConfig:
    batch_size: int
    gradient_accumulation: int
    max_steps: int
    num_workers: int
    seed: int
    log_interval: int
    checkpoint_interval: int

    def __post_init__(self) -> None:
        positive = (
            self.batch_size,
            self.gradient_accumulation,
            self.max_steps,
            self.log_interval,
            self.checkpoint_interval,
        )
        if any(value <= 0 for value in positive) or self.num_workers < 0:
            raise ValueError(f"Invalid Stage 3 train loop configuration: {self}")


@dataclass(frozen=True)
class Stage3TrainingConfig:
    model: EntropyModelConfig
    optimizer: Stage3OptimizerConfig
    loss: Stage3LossConfig
    train: Stage3TrainLoopConfig


T = TypeVar("T")


def _strict_required_construct(cls: type[T], values: dict[str, Any]) -> T:
    required = {field.name for field in fields(cls)}
    supplied = set(values)
    if supplied != required:
        raise ValueError(
            f"{cls.__name__} requires exactly {sorted(required)}, got {sorted(supplied)}"
        )
    return cls(**values)


def load_stage3_training_config(path: str | Path) -> Stage3TrainingConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    sections = {"model", "optimizer", "loss", "train"}
    if not isinstance(raw, dict) or set(raw) != sections:
        raise ValueError(f"Stage 3 config must contain exactly {sorted(sections)}")
    if not all(isinstance(raw[section], dict) for section in sections):
        raise TypeError("Stage 3 config sections must be mappings")

    model_values = dict(raw["model"])
    model_values["levels"] = tuple(model_values["levels"])
    optimizer_values = dict(raw["optimizer"])
    optimizer_values["betas"] = tuple(optimizer_values["betas"])
    return Stage3TrainingConfig(
        model=_strict_required_construct(EntropyModelConfig, model_values),
        optimizer=_strict_required_construct(Stage3OptimizerConfig, optimizer_values),
        loss=_strict_required_construct(Stage3LossConfig, dict(raw["loss"])),
        train=_strict_required_construct(Stage3TrainLoopConfig, dict(raw["train"])),
    )
