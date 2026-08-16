from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from entropy_model.config_row_group import RowGroupEntropyModelConfig
from training.config_ar import OptimizerConfig, TrainLoopConfig


@dataclass(frozen=True)
class RowGroupARTrainingConfig:
    model: RowGroupEntropyModelConfig
    optimizer: OptimizerConfig
    train: TrainLoopConfig


T = TypeVar("T")


def _strict_construct(cls: type[T], values: dict[str, Any]) -> T:
    allowed = {field.name for field in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


def load_row_group_ar_training_config(
    path: str | Path,
) -> RowGroupARTrainingConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or set(raw) != {"model", "optimizer", "train"}:
        raise ValueError("AR config must contain exactly model, optimizer, and train mappings")
    if not all(isinstance(raw[key], dict) for key in raw):
        raise TypeError("AR config sections must be mappings")

    model_values = dict(raw["model"])
    if "levels" in model_values:
        model_values["levels"] = tuple(model_values["levels"])
    optimizer_values = dict(raw["optimizer"])
    if "betas" in optimizer_values:
        optimizer_values["betas"] = tuple(optimizer_values["betas"])
    return RowGroupARTrainingConfig(
        model=_strict_construct(RowGroupEntropyModelConfig, model_values),
        optimizer=_strict_construct(OptimizerConfig, optimizer_values),
        train=_strict_construct(TrainLoopConfig, dict(raw["train"])),
    )
