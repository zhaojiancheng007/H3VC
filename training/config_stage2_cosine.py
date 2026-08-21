from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from entropy_model.config_categorical import CategoricalEntropyModelConfig


@dataclass(frozen=True)
class Stage2OptimizerConfig:
    betas: tuple[float, float]
    weight_decay: float
    grad_clip_norm: float
    learning_rate: float
    eta_min: float


@dataclass(frozen=True)
class Stage2TrainConfig:
    batch_size: int
    max_steps: int
    num_workers: int
    seed: int
    log_interval: int
    checkpoint_interval: int
    bf16: bool
    checkpoint_policy: str


@dataclass(frozen=True)
class Stage2Config:
    model: CategoricalEntropyModelConfig
    optimizer: Stage2OptimizerConfig
    train: Stage2TrainConfig


T = TypeVar("T")


def _construct(cls: type[T], values: dict[str, Any]) -> T:
    expected = {field.name for field in fields(cls)}
    if set(values) != expected:
        raise ValueError(
            f"{cls.__name__} requires exactly {sorted(expected)}, "
            f"got {sorted(values)}"
        )
    return cls(**values)


def learning_rate_at_step(
    config: Stage2OptimizerConfig, max_steps: int, step: int
) -> float:
    if max_steps <= 1:
        raise ValueError(f"max_steps must exceed one, got {max_steps}")
    if step < 1 or step > max_steps:
        raise ValueError(f"step must be in [1, {max_steps}], got {step}")
    progress = (step - 1) / (max_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.eta_min + (config.learning_rate - config.eta_min) * cosine


def stage2_config_dict(config: Stage2Config) -> dict[str, Any]:
    return asdict(config)


def load_stage2_config(path: str | Path) -> Stage2Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or set(raw) != {"model", "optimizer", "train"}:
        raise ValueError("Stage2 config requires model, optimizer, and train")
    optimizer = dict(raw["optimizer"])
    optimizer["betas"] = tuple(optimizer["betas"])
    config = Stage2Config(
        model=_construct(CategoricalEntropyModelConfig, dict(raw["model"])),
        optimizer=_construct(Stage2OptimizerConfig, optimizer),
        train=_construct(Stage2TrainConfig, dict(raw["train"])),
    )
    if min(
        config.optimizer.grad_clip_norm,
        config.train.batch_size,
        config.train.max_steps,
        config.train.num_workers,
        config.train.log_interval,
        config.train.checkpoint_interval,
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
    return config
