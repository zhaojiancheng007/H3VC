from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from entropy_model.config_categorical import CategoricalEntropyModelConfig


@dataclass(frozen=True)
class Stage3OptimizerConfig:
    tokenizer_learning_rate: float
    entropy_learning_rate: float
    discriminator_learning_rate: float
    betas: tuple[float, float]
    grad_clip_norm: float


@dataclass(frozen=True)
class Stage3LossConfig:
    mse_weight: float
    lpips_weight: float
    commitment_weight: float
    initial_rate_weight: float
    base_rate_weight: float
    final_rate_weight: float
    rate_warmup_end_step: int
    rate_ramp_start_step: int
    rate_ramp_end_step: int
    initial_temperature: float
    final_temperature: float
    gan_start_step: int
    gan_weight: float


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


def load_stage3_config(path: str | Path) -> Stage3Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    expected = {"model", "optimizer", "loss", "train"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError(f"Stage3 config requires exactly {sorted(expected)}")
    optimizer = dict(raw["optimizer"])
    optimizer["betas"] = tuple(optimizer["betas"])
    config = Stage3Config(
        model=_construct(CategoricalEntropyModelConfig, dict(raw["model"])),
        optimizer=_construct(Stage3OptimizerConfig, optimizer),
        loss=_construct(Stage3LossConfig, dict(raw["loss"])),
        train=_construct(Stage3TrainConfig, dict(raw["train"])),
    )
    positive = (
        config.optimizer.tokenizer_learning_rate,
        config.optimizer.entropy_learning_rate,
        config.optimizer.discriminator_learning_rate,
        config.optimizer.grad_clip_norm,
        config.loss.mse_weight,
        config.loss.lpips_weight,
        config.loss.commitment_weight,
        config.loss.initial_rate_weight,
        config.loss.base_rate_weight,
        config.loss.final_rate_weight,
        config.loss.initial_temperature,
        config.loss.final_temperature,
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
    if not (
        0 < config.loss.rate_warmup_end_step
        < config.loss.rate_ramp_start_step
        < config.loss.rate_ramp_end_step
        <= config.train.max_steps
    ):
        raise ValueError("rate schedule steps must be ordered within Stage3 training")
    if not 0 <= config.loss.gan_start_step < config.train.max_steps:
        raise ValueError("gan_start_step must occur during Stage3 training")
    return config
