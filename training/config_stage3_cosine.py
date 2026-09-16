from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from entropy_model.config_categorical import CategoricalEntropyModelConfig


@dataclass(frozen=True)
class Stage3OptimizerConfig:
    betas: tuple[float, float]
    grad_clip_norm: float
    tokenizer_learning_rate: float
    tokenizer_eta_min: float
    entropy_learning_rate: float
    entropy_eta_min: float
    weight_decay: float = 0.0


@dataclass(frozen=True)
class Stage3LossConfig:
    mse_weight: float
    lpips_weight: float
    commitment_weight: float
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


def cosine_learning_rates_at_step(
    config: Stage3Config, step: int
) -> tuple[float, float]:
    if step < 1 or step > config.train.max_steps:
        raise ValueError(f"step must be in [1, {config.train.max_steps}], got {step}")
    progress = (step - 1) / (config.train.max_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (
        config.optimizer.tokenizer_eta_min
        + (config.optimizer.tokenizer_learning_rate - config.optimizer.tokenizer_eta_min) * cosine,
        config.optimizer.entropy_eta_min
        + (config.optimizer.entropy_learning_rate - config.optimizer.entropy_eta_min) * cosine,
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
    loss = dict(raw["loss"])
    optimizer["betas"] = tuple(optimizer["betas"])
    config = Stage3Config(
        model=_construct(CategoricalEntropyModelConfig, dict(raw["model"])),
        optimizer=_construct(Stage3OptimizerConfig, optimizer),
        loss=_construct(Stage3LossConfig, loss),
        train=_construct(Stage3TrainConfig, dict(raw["train"])),
    )
    positive = (
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
    if config.optimizer.weight_decay != 0:
        raise ValueError("Stage3 uses Adam with weight_decay=0")
    if not (
        config.optimizer.tokenizer_learning_rate >= config.optimizer.tokenizer_eta_min >= 0
        and config.optimizer.entropy_learning_rate >= config.optimizer.entropy_eta_min >= 0
    ):
        raise ValueError("Cosine LR requires each initial rate to be at least eta_min >= 0")
    if config.train.max_steps <= 1:
        raise ValueError("Cosine LR requires max_steps > 1")
    return config
