from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalGlobalMemoryEntropyConfig:
    codebook_size: int = 4096
    codebook_dim: int = 24
    window_size: int = 8
    d_model: int = 384
    num_layers: int = 8
    num_heads: int = 6
    ffn_dim: int = 1024
    max_latent_frames: int = 32
    memory_tokens: int = 512
    summary_tokens: int = 32
    global_tokens: int = 32
    global_read_layers: int = 2

    def __post_init__(self) -> None:
        if self.codebook_size != 4096 or self.codebook_dim not in {16, 24}:
            raise ValueError("Expected a 4096x16 or 4096x24 VQ codebook")
        if self.window_size <= 0 or self.window_size % 2:
            raise ValueError("window_size must be a positive even integer")
        if self.d_model <= 0 or self.d_model % self.num_heads:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if min(self.num_layers, self.ffn_dim, self.max_latent_frames) <= 0:
            raise ValueError("Transformer dimensions must be positive")
        if self.memory_tokens != 512:
            raise ValueError("The FIFO memory must contain exactly 512 tokens")
        if self.summary_tokens != 32 or self.global_tokens != 32:
            raise ValueError("Summary and global readout must both contain 32 tokens")
        if self.memory_tokens % self.summary_tokens:
            raise ValueError("memory_tokens must be divisible by summary_tokens")
        if self.global_read_layers != 2:
            raise ValueError("The global-memory reader must contain exactly two layers")

    @property
    def memory_frames(self) -> int:
        return self.memory_tokens // self.summary_tokens


__all__ = ["LocalGlobalMemoryEntropyConfig"]
