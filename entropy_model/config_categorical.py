from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CategoricalEntropyModelConfig:
    codebook_size: int = 4096
    codebook_dim: int = 16
    window_size: int = 8
    d_model: int = 384
    num_layers: int = 8
    num_heads: int = 6
    ffn_dim: int = 1024
    max_latent_frames: int = 9

    def __post_init__(self) -> None:
        if self.codebook_size != 4096 or self.codebook_dim not in {16, 24}:
            raise ValueError(
                "The unified entropy model requires a 4096x16 or 4096x24 codebook"
            )
        if self.window_size <= 0 or self.window_size % 2:
            raise ValueError("window_size must be a positive even integer")
        if min(self.d_model, self.num_layers, self.ffn_dim) <= 0:
            raise ValueError("Transformer dimensions must be positive")
        if self.num_heads <= 0 or self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.max_latent_frames <= 0:
            raise ValueError("max_latent_frames must be positive")
