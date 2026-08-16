from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

import torch

from .config import EntropyModelConfig
from .model import CosmosFSQEntropyModel


CHECKPOINT_VERSION = 1


def save_entropy_checkpoint(
    path: str | Path,
    model: CosmosFSQEntropyModel,
    *,
    step: int,
    optimizer_state: dict | None = None,
) -> None:
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "step": step,
        "model_config": asdict(model.config),
        "model": model.state_dict(),
    }
    if optimizer_state is not None:
        payload["optimizer"] = optimizer_state
    torch.save(payload, path)


def load_entropy_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[CosmosFSQEntropyModel, dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {payload.get('version')}, expected {CHECKPOINT_VERSION}"
        )
    config_values = dict(payload["model_config"])
    config_values["levels"] = tuple(config_values["levels"])
    model = CosmosFSQEntropyModel(EntropyModelConfig(**config_values))
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model, payload


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
