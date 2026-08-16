from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BlockTensor:
    values: torch.Tensor
    batch_size: int
    latent_frames: int
    height: int
    width: int
    windows_y: int
    windows_x: int

    @property
    def num_windows(self) -> int:
        return self.windows_y * self.windows_x


class FourGroupBlockLayout:
    """Maps [B,T,H,W,C] grids to time-major, four-color window blocks."""

    num_groups = 4

    def __init__(self, window_size: int = 8) -> None:
        if window_size <= 0 or window_size % 2:
            raise ValueError(f"window_size must be a positive even integer, got {window_size}")
        self.window_size = window_size
        positions: list[list[int]] = []
        for group in range(self.num_groups):
            y_parity, x_parity = divmod(group, 2)
            positions.append(
                [
                    y * window_size + x
                    for y in range(y_parity, window_size, 2)
                    for x in range(x_parity, window_size, 2)
                ]
            )
        lengths = {len(group_positions) for group_positions in positions}
        if len(lengths) != 1:
            raise RuntimeError(f"Four-color groups have unequal sizes: {sorted(lengths)}")
        self._positions = tuple(tuple(values) for values in positions)
        grouped_order = [index for values in self._positions for index in values]
        inverse = [0] * (window_size * window_size)
        for grouped_index, flat_index in enumerate(grouped_order):
            inverse[flat_index] = grouped_index
        self._inverse = tuple(inverse)

    @property
    def sites_per_block(self) -> int:
        return len(self._positions[0])

    @property
    def positions(self) -> tuple[tuple[int, ...], ...]:
        return self._positions

    def _validate_grid(self, grid: torch.Tensor) -> tuple[int, int, int, int, int]:
        if grid.ndim != 5:
            raise ValueError(f"Expected grid [B,T,H,W,C], got {tuple(grid.shape)}")
        batch, frames, height, width, channels = map(int, grid.shape)
        if min(batch, frames, height, width, channels) <= 0:
            raise ValueError(f"Grid dimensions must be positive, got {tuple(grid.shape)}")
        if height % self.window_size or width % self.window_size:
            raise ValueError(
                f"Latent size {(height, width)} must be divisible by window_size={self.window_size}"
            )
        return batch, frames, height, width, channels

    def to_blocks(self, grid: torch.Tensor) -> BlockTensor:
        batch, frames, height, width, channels = self._validate_grid(grid)
        windows_y = height // self.window_size
        windows_x = width // self.window_size
        windows = grid.reshape(
            batch,
            frames,
            windows_y,
            self.window_size,
            windows_x,
            self.window_size,
            channels,
        ).permute(0, 2, 4, 1, 3, 5, 6)
        flat = windows.reshape(
            batch, windows_y, windows_x, frames, self.window_size**2, channels
        )
        positions = torch.tensor(self._positions, dtype=torch.long, device=grid.device)
        grouped = flat[:, :, :, :, positions, :]
        values = grouped.reshape(
            batch * windows_y * windows_x,
            frames * self.num_groups,
            self.sites_per_block,
            channels,
        )
        return BlockTensor(values, batch, frames, height, width, windows_y, windows_x)

    def from_blocks(self, blocks: BlockTensor) -> torch.Tensor:
        values = blocks.values
        expected = (
            blocks.batch_size * blocks.num_windows,
            blocks.latent_frames * self.num_groups,
            self.sites_per_block,
        )
        if values.ndim != 4 or tuple(values.shape[:3]) != expected:
            raise ValueError(
                f"Expected block prefix {expected}, got {tuple(values.shape)}"
            )
        channels = int(values.shape[-1])
        grouped = values.reshape(
            blocks.batch_size,
            blocks.windows_y,
            blocks.windows_x,
            blocks.latent_frames,
            self.window_size**2,
            channels,
        )
        inverse = torch.tensor(self._inverse, dtype=torch.long, device=values.device)
        flat = grouped[..., inverse, :]
        windows = flat.reshape(
            blocks.batch_size,
            blocks.windows_y,
            blocks.windows_x,
            blocks.latent_frames,
            self.window_size,
            self.window_size,
            channels,
        )
        return windows.permute(0, 3, 1, 4, 2, 5, 6).reshape(
            blocks.batch_size,
            blocks.latent_frames,
            blocks.height,
            blocks.width,
            channels,
        )

    def coordinates(
        self, latent_frames: int, *, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if latent_frames <= 0:
            raise ValueError(f"latent_frames must be positive, got {latent_frames}")
        positions = torch.tensor(self._positions, dtype=torch.long, device=device)
        local_y = positions // self.window_size
        local_x = positions % self.window_size
        time = torch.arange(latent_frames, device=device)[:, None, None]
        group = torch.arange(self.num_groups, device=device)[None, :, None]
        time = time.expand(-1, self.num_groups, self.sites_per_block).reshape(-1, self.sites_per_block)
        group = group.expand(latent_frames, -1, self.sites_per_block).reshape(-1, self.sites_per_block)
        local_y = local_y.unsqueeze(0).expand(latent_frames, -1, -1).reshape(-1, self.sites_per_block)
        local_x = local_x.unsqueeze(0).expand(latent_frames, -1, -1).reshape(-1, self.sites_per_block)
        return time, group, local_y, local_x
