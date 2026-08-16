from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from entropy_model.config_categorical import CategoricalEntropyModelConfig


@dataclass(frozen=True)
class Stage3OptimizerConfig:
    discriminator_learning_rate: float
    betas: tuple[float, float]
    grad_clip_norm: float
    tokenizer_learning_rate_schedule: tuple[tuple[int, float], ...]
    entropy_learning_rate_schedule: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class Stage3LossConfig:
    mse_weight: float
    lpips_weight: float
    commitment_weight: float
    gan_start_step: int
    gan_weight: float
    rate_weight: float
    temperature: float


@dataclass(frozen=True)
class Stage3TrainConfig:
    batch_size: int
    max_steps: int
    num_workers: int
    seed: int
    log_interval: int
    checkpoint_interval: int
    bf16: bool


@dataclass(frozen=True)
class Stage3Config:
    model: CategoricalEntropyModelConfig
    optimizer: Stage3OptimizerConfig
    loss: Stage3LossConfig
    train: Stage3TrainConfig


T = TypeVar("T")


def _construct(cls: type[T], values: dict[str, Any]) -> T:
    expected = {field.name for field in fields(cls)}
    if set(values) != expected:
        raise ValueError(
            f"{cls.__name__} requires exactly {sorted(expected)}, "
            f"got {sorted(values)}"
        )
    return cls(**values)


def _scheduled_value(schedule: tuple[tuple[int, float], ...], step: int) -> float:
    milestones = tuple(pair[0] for pair in schedule)
    return schedule[bisect_right(milestones, step) - 1][1]


def learning_rates_at_step(
    config: Stage3OptimizerConfig, step: int
) -> tuple[float, float]:
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    return (
        _scheduled_value(config.tokenizer_learning_rate_schedule, step),
        _scheduled_value(config.entropy_learning_rate_schedule, step),
    )


def stage3_config_dict(config: Stage3Config) -> dict[str, Any]:
    return asdict(config)


def load_stage3_config(path: str | Path) -> Stage3Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    expected = {"model", "optimizer", "loss", "train"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError(f"Stage3 config requires exactly {sorted(expected)}")
    optimizer = dict(raw["optimizer"])
    optimizer["betas"] = tuple(optimizer["betas"])
    for name in (
        "tokenizer_learning_rate_schedule",
        "entropy_learning_rate_schedule",
    ):
        if name in optimizer:
            optimizer[name] = tuple(
                (int(step), float(rate)) for step, rate in optimizer[name]
            )
    config = Stage3Config(
        model=_construct(CategoricalEntropyModelConfig, dict(raw["model"])),
        optimizer=_construct(Stage3OptimizerConfig, optimizer),
        loss=_construct(Stage3LossConfig, dict(raw["loss"])),
        train=_construct(Stage3TrainConfig, dict(raw["train"])),
    )
    positive = (
        config.optimizer.discriminator_learning_rate,
        config.optimizer.grad_clip_norm,
        config.loss.mse_weight,
        config.loss.lpips_weight,
        config.loss.commitment_weight,
        config.loss.rate_weight,
        config.loss.temperature,
        config.train.batch_size,
        config.train.max_steps,
        config.train.num_workers,
        config.train.log_interval,
        config.train.checkpoint_interval,
    )
    if min(positive) <= 0:
        raise ValueError(f"Stage3 values must be positive: {config}")
    if config.loss.gan_weight < 0:
        raise ValueError("gan_weight must be non-negative")
    tokenizer_schedule = config.optimizer.tokenizer_learning_rate_schedule
    entropy_schedule = config.optimizer.entropy_learning_rate_schedule
    for name, schedule in (
        ("tokenizer", tokenizer_schedule),
        ("entropy", entropy_schedule),
    ):
        steps = tuple(pair[0] for pair in schedule)
        rates = tuple(pair[1] for pair in schedule)
        if not steps or steps[0] != 0 or any(
            left >= right for left, right in zip(steps, steps[1:])
        ):
            raise ValueError(f"Stage3 {name} LR steps must start at 0 and increase")
        if any(rate <= 0 for rate in rates) or steps[-1] >= config.train.max_steps:
            raise ValueError(f"Invalid Stage3 {name} learning-rate schedule")
    if tuple(pair[0] for pair in tokenizer_schedule) != tuple(
        pair[0] for pair in entropy_schedule
    ):
        raise ValueError("Tokenizer and entropy LR milestones must match")
    if not 0 <= config.loss.gan_start_step < config.train.max_steps:
        raise ValueError("gan_start_step must occur during Stage3 training")
    return config
