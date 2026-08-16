from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any

import torch

from entropy_model import CosmosFSQEntropyModel
from entropy_model.checkpoint import CHECKPOINT_VERSION


@dataclass(frozen=True)
class DDPTrainingState:
    step: int
    epoch: int
    batch_in_epoch: int
    max_steps: int
    world_size: int
    global_batch_size: int

    def __post_init__(self) -> None:
        if self.step < 0 or self.epoch < 0 or self.batch_in_epoch < 0:
            raise ValueError(f"Training counters must be non-negative: {self}")
        if self.max_steps <= 0 or self.step > self.max_steps:
            raise ValueError(f"Invalid max_steps/step pair: {self}")
        if self.world_size <= 0 or self.global_batch_size <= 0:
            raise ValueError(f"Invalid distributed training state: {self}")


def save_ddp_training_checkpoint(
    path: str | Path,
    model: CosmosFSQEntropyModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    state: DDPTrainingState,
    rng_states: list[dict[str, Any]],
) -> None:
    if len(rng_states) != state.world_size:
        raise ValueError(
            f"Expected {state.world_size} RNG states, received {len(rng_states)}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    payload = {
        "version": CHECKPOINT_VERSION,
        "step": state.step,
        "model_config": asdict(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "training_state": asdict(state),
        "rng_states": rng_states,
    }
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def load_ddp_training_checkpoint(
    path: str | Path, *, device: str | torch.device
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    required = {
        "version",
        "step",
        "model_config",
        "model",
        "optimizer",
        "scheduler",
        "training_state",
        "rng_states",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"DDP checkpoint is missing fields: {sorted(missing)}")
    if payload["version"] != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {payload['version']}, expected {CHECKPOINT_VERSION}"
        )
    state = DDPTrainingState(**payload["training_state"])
    if payload["step"] != state.step:
        raise ValueError(
            f"Top-level step {payload['step']} disagrees with training_state.step={state.step}"
        )
    if len(payload["rng_states"]) != state.world_size:
        raise ValueError(
            f"Checkpoint has {len(payload['rng_states'])} RNG states for world_size={state.world_size}"
        )
    return payload
