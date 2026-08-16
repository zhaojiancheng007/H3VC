from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RowGroupTensor:
    values: torch.Tensor
    batch_size: int
    latent_frames: int
    height: int
    width: int
    rows_per_group: int

    @property
    def groups_per_frame(self) -> int:
        return self.height // self.rows_per_group


class RowGroupLayout:
    """Maps [B,T,H,W,C] to time-major groups of complete latent rows."""

    def __init__(self, rows_per_group: int = 4) -> None:
        if rows_per_group <= 0:
            raise ValueError(f"rows_per_group must be positive, got {rows_per_group}")
        self.rows_per_group = rows_per_group

    def _validate_grid(self, grid: torch.Tensor) -> tuple[int, int, int, int, int]:
        if grid.ndim != 5:
            raise ValueError(f"Expected grid [B,T,H,W,C], got {tuple(grid.shape)}")
        batch, frames, height, width, channels = map(int, grid.shape)
        if min(batch, frames, height, width, channels) <= 0:
            raise ValueError(f"Grid dimensions must be positive, got {tuple(grid.shape)}")
        if height % self.rows_per_group:
            raise ValueError(
                f"Latent height {height} must be divisible by "
                f"rows_per_group={self.rows_per_group}"
            )
        return batch, frames, height, width, channels

    def to_blocks(self, grid: torch.Tensor) -> RowGroupTensor:
        batch, frames, height, width, channels = self._validate_grid(grid)
        groups = height // self.rows_per_group
        values = grid.reshape(
            batch,
            frames,
            groups,
            self.rows_per_group * width,
            channels,
        ).reshape(
            batch,
            frames * groups,
            self.rows_per_group * width,
            channels,
        )
        return RowGroupTensor(
            values=values,
            batch_size=batch,
            latent_frames=frames,
            height=height,
            width=width,
            rows_per_group=self.rows_per_group,
        )

    def from_blocks(self, blocks: RowGroupTensor) -> torch.Tensor:
        values = blocks.values
        expected = (
            blocks.batch_size,
            blocks.latent_frames * blocks.groups_per_frame,
            blocks.rows_per_group * blocks.width,
        )
        if values.ndim != 4 or tuple(values.shape[:3]) != expected:
            raise ValueError(f"Expected block prefix {expected}, got {tuple(values.shape)}")
        channels = int(values.shape[-1])
        return values.reshape(
            blocks.batch_size,
            blocks.latent_frames,
            blocks.height,
            blocks.width,
            channels,
        )
