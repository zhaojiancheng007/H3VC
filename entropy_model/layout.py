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
    temporal_window_size: int

    @property
    def num_windows(self) -> int:
        return self.windows_y * self.windows_x

    @property
    def ranks_per_window(self) -> int:
        return self.latent_frames * 4


MIXED_PREFIX_K_PROBABILITIES = {1: 1.0 / 6.0, 2: 1.0 / 6.0, 3: 1.0 / 6.0, 4: 0.5}


def training_k_probabilities(fixed_k: int, mixed_prefix: bool) -> dict[int, float]:
    if fixed_k not in {1, 2, 3, 4}:
        raise ValueError(f"fixed_k must be one of 1, 2, 3, 4, got {fixed_k}")
    if mixed_prefix:
        return dict(MIXED_PREFIX_K_PROBABILITIES)
    return {fixed_k: 1.0}


def training_k_for_step(
    fixed_k: int, mixed_prefix: bool, seed: int, step: int
) -> int:
    if fixed_k not in {1, 2, 3, 4}:
        raise ValueError(f"fixed_k must be one of 1, 2, 3, 4, got {fixed_k}")
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    if not mixed_prefix:
        return fixed_k

    value = (seed + step * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
    value ^= value >> 31
    uniform = (value >> 11) / float(1 << 53)
    if uniform < 1.0 / 6.0:
        return 1
    if uniform < 2.0 / 6.0:
        return 2
    if uniform < 3.0 / 6.0:
        return 3
    return 4


class SpatiotemporalWindowLayout:
    """Maps same-position 8x8 windows to one diagonal causal rank sequence."""

    num_groups = 4

    def __init__(self, window_size: int, temporal_window_size: int) -> None:
        if window_size <= 0 or window_size % 2:
            raise ValueError(
                f"window_size must be a positive even integer, got {window_size}"
            )
        if temporal_window_size not in {1, 2, 3, 4}:
            raise ValueError(
                "temporal_window_size must be one of 1, 2, 3, 4, "
                f"got {temporal_window_size}"
            )
        self.window_size = window_size
        self.temporal_window_size = temporal_window_size
        positions: list[tuple[int, ...]] = []
        for group in range(self.num_groups):
            y_parity, x_parity = divmod(group, 2)
            positions.append(
                tuple(
                    y * window_size + x
                    for y in range(y_parity, window_size, 2)
                    for x in range(x_parity, window_size, 2)
                )
            )
        self._positions = tuple(positions)

    @property
    def sites_per_block(self) -> int:
        return len(self._positions[0])

    @property
    def positions(self) -> tuple[tuple[int, ...], ...]:
        return self._positions

    def rank_order(self, latent_frames: int) -> tuple[tuple[int, int], ...]:
        if latent_frames <= 0:
            raise ValueError(f"latent_frames must be positive, got {latent_frames}")
        order: list[tuple[int, int]] = []
        for start in range(0, latent_frames, self.temporal_window_size):
            length = min(self.temporal_window_size, latent_frames - start)
            local = sorted(
                (
                    (local_time, group)
                    for local_time in range(length)
                    for group in range(self.num_groups)
                ),
                key=lambda item: (item[0] + item[1], -item[0]),
            )
            order.extend((start + local_time, group) for local_time, group in local)
        return tuple(order)

    def _flat_order(self, latent_frames: int) -> tuple[int, ...]:
        return tuple(
            time * self.window_size**2 + position
            for time, group in self.rank_order(latent_frames)
            for position in self._positions[group]
        )

    def _validate_grid(self, grid: torch.Tensor) -> tuple[int, int, int, int, int]:
        if grid.ndim != 5:
            raise ValueError(f"Expected grid [B,T,H,W,C], got {tuple(grid.shape)}")
        batch, frames, height, width, channels = map(int, grid.shape)
        if min(batch, frames, height, width, channels) <= 0:
            raise ValueError(f"Grid dimensions must be positive, got {tuple(grid.shape)}")
        if height % self.window_size or width % self.window_size:
            raise ValueError(
                f"Latent size {(height, width)} must be divisible by "
                f"window_size={self.window_size}"
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
            batch,
            windows_y,
            windows_x,
            frames * self.window_size**2,
            channels,
        )
        order = torch.tensor(
            self._flat_order(frames), dtype=torch.long, device=grid.device
        )
        grouped = flat[..., order, :].reshape(
            batch,
            windows_y,
            windows_x,
            frames * self.num_groups,
            self.sites_per_block,
            channels,
        )
        values = grouped.reshape(
            batch * windows_y * windows_x,
            frames * self.num_groups,
            self.sites_per_block,
            channels,
        )
        return BlockTensor(
            values=values,
            batch_size=batch,
            latent_frames=frames,
            height=height,
            width=width,
            windows_y=windows_y,
            windows_x=windows_x,
            temporal_window_size=self.temporal_window_size,
        )

    def from_blocks(self, blocks: BlockTensor) -> torch.Tensor:
        if blocks.temporal_window_size != self.temporal_window_size:
            raise ValueError("Block metadata uses a different temporal window size")
        expected = (
            blocks.batch_size * blocks.num_windows,
            blocks.ranks_per_window,
            self.sites_per_block,
        )
        if blocks.values.ndim != 4 or tuple(blocks.values.shape[:3]) != expected:
            raise ValueError(
                f"Expected block prefix {expected}, got {tuple(blocks.values.shape)}"
            )
        channels = int(blocks.values.shape[-1])
        grouped = blocks.values.reshape(
            blocks.batch_size,
            blocks.windows_y,
            blocks.windows_x,
            blocks.latent_frames * self.window_size**2,
            channels,
        )
        flat_order = self._flat_order(blocks.latent_frames)
        inverse = [0] * len(flat_order)
        for grouped_index, flat_index in enumerate(flat_order):
            inverse[flat_index] = grouped_index
        inverse_tensor = torch.tensor(
            inverse, dtype=torch.long, device=blocks.values.device
        )
        flat = grouped[..., inverse_tensor, :].reshape(
            blocks.batch_size,
            blocks.windows_y,
            blocks.windows_x,
            blocks.latent_frames,
            self.window_size,
            self.window_size,
            channels,
        )
        return flat.permute(0, 3, 1, 4, 2, 5, 6).reshape(
            blocks.batch_size,
            blocks.latent_frames,
            blocks.height,
            blocks.width,
            channels,
        )

    def coordinates(
        self, latent_frames: int, *, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        order = self.rank_order(latent_frames)
        time = torch.tensor(
            [time for time, _ in order], dtype=torch.long, device=device
        )
        group = torch.tensor(
            [group for _, group in order], dtype=torch.long, device=device
        )
        positions = torch.tensor(self._positions, dtype=torch.long, device=device)
        local_y = positions[group] // self.window_size
        local_x = positions[group] % self.window_size
        time = time[:, None].expand(-1, self.sites_per_block)
        group = group[:, None].expand(-1, self.sites_per_block)
        return time, group, local_y, local_x


__all__ = [
    "BlockTensor",
    "MIXED_PREFIX_K_PROBABILITIES",
    "SpatiotemporalWindowLayout",
    "training_k_probabilities",
    "training_k_for_step",
]
