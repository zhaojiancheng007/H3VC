from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass(frozen=True)
class Stage1OptimizerConfig:
    learning_rate: float
    betas: tuple[float, float]
    grad_clip_norm: float
    lr_decay_interval_steps: int
    lr_decay_factor: float
    lr_decay_stop_step: int

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.grad_clip_norm <= 0:
            raise ValueError(f"Invalid Stage 1 optimizer configuration: {self}")
        if len(self.betas) != 2 or not all(0 <= beta < 1 for beta in self.betas):
            raise ValueError(f"Invalid Adam betas: {self.betas}")
        if self.lr_decay_interval_steps <= 0:
            raise ValueError("lr_decay_interval_steps must be positive")
        if not 0 < self.lr_decay_factor < 1:
            raise ValueError("lr_decay_factor must be in (0, 1)")
        if (
            self.lr_decay_stop_step <= 0
            or self.lr_decay_stop_step % self.lr_decay_interval_steps
        ):
            raise ValueError(
                "lr_decay_stop_step must be a positive multiple of "
                "lr_decay_interval_steps"
            )


@dataclass(frozen=True)
class Stage1TrainLoopConfig:
    batch_size: int
    max_steps: int
    num_workers: int
    seed: int
    log_interval: int

    def __post_init__(self) -> None:
        positive = (
            self.batch_size,
            self.max_steps,
            self.num_workers,
            self.log_interval,
        )
        if any(value <= 0 for value in positive):
            raise ValueError(f"Invalid Stage 1 train loop configuration: {self}")


@dataclass(frozen=True)
class Stage1TrainingConfig:
    optimizer: Stage1OptimizerConfig
    train: Stage1TrainLoopConfig


T = TypeVar("T")


def _strict_construct(cls: type[T], values: dict[str, Any]) -> T:
    required = {field.name for field in fields(cls)}
    supplied = set(values)
    if supplied != required:
        raise ValueError(
            f"{cls.__name__} requires exactly {sorted(required)}, "
            f"got {sorted(supplied)}"
        )
    return cls(**values)


def load_stage1_training_config(path: str | Path) -> Stage1TrainingConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    sections = {"optimizer", "train"}
    if not isinstance(raw, dict) or set(raw) != sections:
        raise ValueError(f"Stage 1 config must contain exactly {sorted(sections)}")
    if not all(isinstance(raw[section], dict) for section in sections):
        raise TypeError("Stage 1 config sections must be mappings")

    optimizer_values = dict(raw["optimizer"])
    optimizer_values["betas"] = tuple(optimizer_values["betas"])
    return Stage1TrainingConfig(
        optimizer=_strict_construct(Stage1OptimizerConfig, optimizer_values),
        train=_strict_construct(Stage1TrainLoopConfig, dict(raw["train"])),
    )
