from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from entropy_model.config_local_global_memory import LocalGlobalMemoryEntropyConfig
from training.config_stage2_cosine import (
    Stage2OptimizerConfig,
    load_stage2_config as load_local_config,
    Stage2TrainConfig,
    _construct,
    learning_rate_at_step,
)


@dataclass(frozen=True)
class LocalGlobalMemoryStage2Config:
    model: LocalGlobalMemoryEntropyConfig
    optimizer: Stage2OptimizerConfig
    train: Stage2TrainConfig


def stage2_config_dict(config: LocalGlobalMemoryStage2Config) -> dict[str, Any]:
    return asdict(config)


def load_stage2_config(path: str | Path) -> LocalGlobalMemoryStage2Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or set(raw) != {"model", "optimizer", "train"}:
        raise ValueError("Stage2 config requires model, optimizer, and train")
    if "memory_tokens" not in raw["model"]:
        return load_local_config(path)
    optimizer = dict(raw["optimizer"])
    optimizer["betas"] = tuple(optimizer["betas"])
    config = LocalGlobalMemoryStage2Config(
        model=_construct(LocalGlobalMemoryEntropyConfig, dict(raw["model"])),
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
        raise ValueError(f"Unknown checkpoint policy: {config.train.checkpoint_policy}")
    return config


__all__ = [
    "LocalGlobalMemoryStage2Config",
    "learning_rate_at_step",
    "load_stage2_config",
    "stage2_config_dict",
]
