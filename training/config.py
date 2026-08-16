from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizerDataConfig:
    num_frames: int = 33
    resize_short_side: int = 512
    crop_size: int = 256
