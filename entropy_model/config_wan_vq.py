from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WanVQEntropyModelConfig:
    codebook_size: int = 4096
    codebook_dim: int = 32
    window_size: int = 8
    d_model: int = 384
    num_layers: int = 8
    num_heads: int = 6
    ffn_dim: int = 1024
    max_latent_frames: int = 9

    @property
    def num_bits(self) -> int:
        return self.codebook_size.bit_length() - 1

    def __post_init__(self) -> None:
        if self.codebook_size != 4096 or self.codebook_dim != 32:
            raise ValueError("WanVQ Stage 2 requires a 4096x32 codebook")
        if self.codebook_size & (self.codebook_size - 1):
            raise ValueError("codebook_size must be a power of two")
        if self.window_size <= 0 or self.window_size % 2:
            raise ValueError("window_size must be a positive even integer")
        if self.d_model <= 0 or self.num_layers <= 0 or self.ffn_dim <= 0:
            raise ValueError("d_model, num_layers, and ffn_dim must be positive")
        if self.num_heads <= 0 or self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.max_latent_frames <= 0:
            raise ValueError("max_latent_frames must be positive")
