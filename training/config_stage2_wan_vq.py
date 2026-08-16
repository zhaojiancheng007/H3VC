from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from entropy_model.config_wan_vq import WanVQEntropyModelConfig
from training.config_ar import OptimizerConfig, TrainLoopConfig


def _construct(cls, values: dict[str, Any]):
    allowed = set(cls.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


def load_wan_vq_ar_config(path: str | Path) -> tuple[
    WanVQEntropyModelConfig, OptimizerConfig, TrainLoopConfig
]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or set(raw) != {"model", "optimizer", "train"}:
        raise ValueError("WanVQ AR config must contain model, optimizer, and train")
    model = _construct(WanVQEntropyModelConfig, dict(raw["model"]))
    optimizer = _construct(OptimizerConfig, dict(raw["optimizer"]))
    train = _construct(TrainLoopConfig, dict(raw["train"]))
    return model, optimizer, train
