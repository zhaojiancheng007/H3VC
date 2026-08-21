from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass(frozen=True)
class Stage1OptimizerConfig:
    betas: tuple[float, float]
    grad_clip_norm: float
    learning_rate: float
    eta_min: float

    def __post_init__(self) -> None:
        if self.grad_clip_norm <= 0:
            raise ValueError(f"Invalid Stage 1 optimizer configuration: {self}")
        if len(self.betas) != 2 or not all(0 <= beta < 1 for beta in self.betas):
            raise ValueError(f"Invalid Adam betas: {self.betas}")
        if not self.learning_rate > self.eta_min >= 0:
            raise ValueError("Cosine LR requires learning_rate > eta_min >= 0")


@dataclass(frozen=True)
class Stage1TrainLoopConfig:
    batch_size: int
    max_steps: int
    num_workers: int
    seed: int
    log_interval: int
    checkpoint_interval: int
    checkpoint_policy: str

    def __post_init__(self) -> None:
        positive = (
            self.batch_size,
            self.max_steps,
            self.num_workers,
            self.log_interval,
            self.checkpoint_interval,
        )
        if any(value <= 0 for value in positive):
            raise ValueError(f"Invalid Stage 1 train loop configuration: {self}")
        if self.checkpoint_policy != "best_lpips":
            raise ValueError(
                f"Unknown Stage 1 checkpoint policy: {self.checkpoint_policy}"
            )


@dataclass(frozen=True)
class Stage1TrainingConfig:
    optimizer: Stage1OptimizerConfig
    train: Stage1TrainLoopConfig


T = TypeVar("T")


def _strict_construct(cls: type[T], values: dict[str, Any]) -> T:
    expected = {field.name for field in fields(cls)}
    if set(values) != expected:
        raise ValueError(
            f"{cls.__name__} requires exactly {sorted(expected)}, "
            f"got {sorted(values)}"
        )
    return cls(**values)


def learning_rate_at_step(
    config: Stage1OptimizerConfig, max_steps: int, step: int
) -> float:
    if max_steps <= 1:
        raise ValueError(f"max_steps must exceed one, got {max_steps}")
    if step < 1 or step > max_steps:
        raise ValueError(f"step must be in [1, {max_steps}], got {step}")
    progress = (step - 1) / (max_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.eta_min + (config.learning_rate - config.eta_min) * cosine


def learning_rate_schedule_dict(config: Stage1OptimizerConfig) -> dict[str, Any]:
    return {
        "type": "cosine_annealing",
        "learning_rate": config.learning_rate,
        "eta_min": config.eta_min,
    }


def stage1_config_dict(config: Stage1TrainingConfig) -> dict[str, Any]:
    return asdict(config)


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
