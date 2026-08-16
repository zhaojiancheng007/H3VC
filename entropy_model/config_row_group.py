from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RowGroupEntropyModelConfig:
    levels: tuple[int, ...] = (8, 8, 8, 5, 5, 5)
    rows_per_group: int = 4
    d_model: int = 384
    num_layers: int = 8
    num_heads: int = 6
    ffn_dim: int = 1024
    max_latent_frames: int = 9
    max_latent_height: int = 136
    max_latent_width: int = 240

    def __post_init__(self) -> None:
        if self.levels != (8, 8, 8, 5, 5, 5):
            raise ValueError(f"Cosmos DV FSQ requires levels (8,8,8,5,5,5), got {self.levels}")
        if self.rows_per_group <= 0:
            raise ValueError("rows_per_group must be positive")
        if self.max_latent_height % self.rows_per_group:
            raise ValueError("max_latent_height must be divisible by rows_per_group")
        positive = (
            self.d_model,
            self.num_layers,
            self.num_heads,
            self.ffn_dim,
            self.max_latent_frames,
            self.max_latent_height,
            self.max_latent_width,
        )
        if any(value <= 0 for value in positive):
            raise ValueError(f"Model dimensions must be positive: {self}")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
