from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from entropy_model.config import EntropyModelConfig
from training.config_stage3 import (
    Stage3LossConfig,
    Stage3OptimizerConfig,
    Stage3TrainLoopConfig,
)


@dataclass(frozen=True)
class Stage3GanConfig:
    start_step: int
    loss_weight: float
    discriminator_learning_rate: float
    betas: tuple[float, float]
    base_channels: int
    num_layers: int

    def __post_init__(self) -> None:
        if self.start_step < 0:
            raise ValueError("GAN start_step must be non-negative")
        if self.loss_weight <= 0 or self.discriminator_learning_rate <= 0:
            raise ValueError("GAN weights and learning rate must be positive")
        if len(self.betas) != 2 or not all(0 <= beta < 1 for beta in self.betas):
            raise ValueError(f"Invalid discriminator Adam betas: {self.betas}")
        if self.base_channels <= 0 or self.num_layers <= 0:
            raise ValueError("PatchGAN dimensions must be positive")


@dataclass(frozen=True)
class Stage3GanTrainingConfig:
    model: EntropyModelConfig
    optimizer: Stage3OptimizerConfig
    loss: Stage3LossConfig
    gan: Stage3GanConfig
    train: Stage3TrainLoopConfig

    def __post_init__(self) -> None:
        if self.gan.start_step >= self.train.max_steps:
            raise ValueError("GAN must start before the final training step")


T = TypeVar("T")


def _strict_construct(cls: type[T], values: dict[str, Any]) -> T:
    required = {field.name for field in fields(cls)}
    if set(values) != required:
        raise ValueError(
            f"{cls.__name__} requires exactly {sorted(required)}, "
            f"got {sorted(values)}"
        )
    return cls(**values)


def load_stage3_gan_training_config(
    path: str | Path,
) -> Stage3GanTrainingConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    sections = {"model", "optimizer", "loss", "gan", "train"}
    if not isinstance(raw, dict) or set(raw) != sections:
        raise ValueError(
            f"Stage 3 GAN config must contain exactly {sorted(sections)}"
        )
    if not all(isinstance(raw[section], dict) for section in sections):
        raise TypeError("Stage 3 GAN config sections must be mappings")

    model_values = dict(raw["model"])
    model_values["levels"] = tuple(model_values["levels"])
    optimizer_values = dict(raw["optimizer"])
    optimizer_values["betas"] = tuple(optimizer_values["betas"])
    gan_values = dict(raw["gan"])
    gan_values["betas"] = tuple(gan_values["betas"])
    return Stage3GanTrainingConfig(
        model=_strict_construct(EntropyModelConfig, model_values),
        optimizer=_strict_construct(Stage3OptimizerConfig, optimizer_values),
        loss=_strict_construct(Stage3LossConfig, dict(raw["loss"])),
        gan=_strict_construct(Stage3GanConfig, gan_values),
        train=_strict_construct(Stage3TrainLoopConfig, dict(raw["train"])),
    )
