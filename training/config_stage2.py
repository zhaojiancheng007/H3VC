from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from entropy_model.config_categorical import CategoricalEntropyModelConfig


@dataclass(frozen=True)
class Stage2OptimizerConfig:
    learning_rate: float
    betas: tuple[float, float]
    weight_decay: float
    grad_clip_norm: float
    warmup_steps: int


@dataclass(frozen=True)
class Stage2TrainConfig:
    batch_size: int
    max_steps: int
    num_workers: int
    seed: int
    log_interval: int
    checkpoint_interval: int
    bf16: bool


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
        config.optimizer.learning_rate,
        config.optimizer.grad_clip_norm,
        config.train.batch_size,
        config.train.max_steps,
        config.train.num_workers,
        config.train.log_interval,
        config.train.checkpoint_interval,
    ) <= 0:
        raise ValueError(f"Stage2 values must be positive: {config}")
    if config.optimizer.weight_decay < 0 or config.optimizer.warmup_steps < 0:
        raise ValueError(f"Invalid Stage2 optimizer values: {config.optimizer}")
    return config
