from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from entropy_model.config_wan_vq import WanVQEntropyModelConfig


@dataclass(frozen=True)
class WanVQStage3OptimizerConfig:
    tokenizer_learning_rate: float
    entropy_learning_rate: float
    discriminator_learning_rate: float
    betas: tuple[float, float]
    grad_clip_norm: float

    def __post_init__(self) -> None:
        if min(
            self.tokenizer_learning_rate,
            self.entropy_learning_rate,
            self.discriminator_learning_rate,
            self.grad_clip_norm,
        ) <= 0:
            raise ValueError("Learning rates and grad_clip_norm must be positive")
        if len(self.betas) != 2 or not all(0 <= value < 1 for value in self.betas):
            raise ValueError(f"Invalid Adam betas: {self.betas}")


@dataclass(frozen=True)
class WanVQStage3LossConfig:
    mse_weight: float
    lpips_weight: float
    commitment_weight: float
    initial_rate_weight: float
    low_rate_weight: float
    initial_temperature: float
    low_rate_temperature: float
    initial_warmup_steps: int
    low_rate_start_step: int
    low_rate_warmup_steps: int
    gan_start_step: int
    gan_weight: float

    def __post_init__(self) -> None:
        positive = (
            self.mse_weight,
            self.lpips_weight,
            self.commitment_weight,
            self.initial_rate_weight,
            self.low_rate_weight,
            self.initial_temperature,
            self.low_rate_temperature,
            self.gan_weight,
        )
        if min(positive) <= 0:
            raise ValueError("All loss weights and temperatures must be positive")
        if min(
            self.initial_warmup_steps,
            self.low_rate_start_step,
            self.low_rate_warmup_steps,
            self.gan_start_step,
        ) < 0:
            raise ValueError("Schedule steps must be non-negative")


@dataclass(frozen=True)
class WanVQStage3TrainConfig:
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
        if min(positive) <= 0 or self.num_workers < 0:
            raise ValueError(f"Invalid training configuration: {self}")


@dataclass(frozen=True)
class WanVQStage3Config:
    model: WanVQEntropyModelConfig
    optimizer: WanVQStage3OptimizerConfig
    loss: WanVQStage3LossConfig
    train: WanVQStage3TrainConfig

    def __post_init__(self) -> None:
        if self.loss.low_rate_start_step + self.loss.low_rate_warmup_steps > self.train.max_steps:
            raise ValueError("Low-rate warmup must finish within training")
        if self.loss.gan_start_step >= self.train.max_steps:
            raise ValueError("GAN must start before training ends")


T = TypeVar("T")


def _strict_construct(cls: type[T], values: dict[str, Any]) -> T:
    required = {field.name for field in fields(cls)}
    if set(values) != required:
        raise ValueError(
            f"{cls.__name__} requires exactly {sorted(required)}, got {sorted(values)}"
        )
    return cls(**values)


def load_wan_vq_stage3_config(path: str | Path) -> WanVQStage3Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    sections = {"model", "optimizer", "loss", "train"}
    if not isinstance(raw, dict) or set(raw) != sections:
        raise ValueError(f"Stage3 config requires exactly {sorted(sections)}")
    if not all(isinstance(raw[section], dict) for section in sections):
        raise TypeError("All Stage3 config sections must be mappings")
    optimizer = dict(raw["optimizer"])
    optimizer["betas"] = tuple(optimizer["betas"])
    return WanVQStage3Config(
        model=_strict_construct(WanVQEntropyModelConfig, dict(raw["model"])),
        optimizer=_strict_construct(WanVQStage3OptimizerConfig, optimizer),
        loss=_strict_construct(WanVQStage3LossConfig, dict(raw["loss"])),
        train=_strict_construct(WanVQStage3TrainConfig, dict(raw["train"])),
    )
