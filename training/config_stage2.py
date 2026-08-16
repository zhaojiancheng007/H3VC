from __future__ import annotations

from bisect import bisect_right
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
    learning_rate_schedule: tuple[tuple[int, float], ...]


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


def learning_rate_at_step(config: Stage2OptimizerConfig, step: int) -> float:
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    steps = tuple(pair[0] for pair in config.learning_rate_schedule)
    index = bisect_right(steps, step) - 1
    return config.learning_rate_schedule[index][1]


def stage2_config_dict(config: Stage2Config) -> dict[str, Any]:
    return asdict(config)


def load_stage2_config(path: str | Path) -> Stage2Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or set(raw) != {"model", "optimizer", "train"}:
        raise ValueError("Stage2 config requires model, optimizer, and train")
    optimizer = dict(raw["optimizer"])
    optimizer["betas"] = tuple(optimizer["betas"])
    optimizer["learning_rate_schedule"] = tuple(
        (int(step), float(rate)) for step, rate in optimizer["learning_rate_schedule"]
    )
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
    if config.optimizer.weight_decay < 0:
        raise ValueError(f"Invalid Stage2 optimizer values: {config.optimizer}")
    schedule = config.optimizer.learning_rate_schedule
    steps = tuple(pair[0] for pair in schedule)
    rates = tuple(pair[1] for pair in schedule)
    if not steps or steps[0] != 0 or any(
        left >= right for left, right in zip(steps, steps[1:])
    ):
        raise ValueError("Stage2 LR steps must start at 0 and increase")
    if any(rate <= 0 for rate in rates) or steps[-1] >= config.train.max_steps:
        raise ValueError("Invalid Stage2 learning-rate schedule")
    if config.train.checkpoint_policy != "best_bpp":
        raise ValueError(
            f"Unknown Stage2 checkpoint policy: {config.train.checkpoint_policy}"
        )
    return config
