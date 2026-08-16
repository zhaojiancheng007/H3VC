from __future__ import annotations

from dataclasses import dataclass


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
