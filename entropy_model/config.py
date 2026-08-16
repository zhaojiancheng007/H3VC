from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EntropyModelConfig:
    levels: tuple[int, ...] = (8, 8, 8, 5, 5, 5)
    window_size: int = 8
    d_model: int = 384
    num_layers: int = 8
    num_heads: int = 6
    ffn_dim: int = 1024
    max_latent_frames: int = 9

    def __post_init__(self) -> None:
        if self.levels != (8, 8, 8, 5, 5, 5):
            raise ValueError(f"Cosmos-DV4x8x8 requires levels (8,8,8,5,5,5), got {self.levels}")
        if self.window_size <= 0 or self.window_size % 2:
            raise ValueError(f"window_size must be a positive even integer, got {self.window_size}")
        if self.d_model <= 0 or self.num_layers <= 0 or self.ffn_dim <= 0:
            raise ValueError("d_model, num_layers, and ffn_dim must be positive")
        if self.num_heads <= 0 or self.d_model % self.num_heads:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by num_heads={self.num_heads}"
            )
        if self.max_latent_frames <= 0:
            raise ValueError("max_latent_frames must be positive")
