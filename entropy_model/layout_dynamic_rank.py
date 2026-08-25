from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DynamicRankBlockTensor:
    values: torch.Tensor
    batch_size: int
    latent_frames: int
    height: int
    width: int
    windows_y: int
    windows_x: int
    rank_to_group: torch.Tensor

    @property
    def num_windows(self) -> int:
        return self.windows_y * self.windows_x


class DynamicRankFourGroupBlockLayout:
    """Maps parity groups to deterministic window/time-dependent coding ranks."""

    num_groups = 4

    def __init__(self, window_size: int = 8) -> None:
        if window_size <= 0 or window_size % 2:
            raise ValueError(
                f"window_size must be a positive even integer, got {window_size}"
            )
        self.window_size = window_size
        positions = []
        for group in range(self.num_groups):
            y_parity, x_parity = divmod(group, 2)
            positions.append(
                [
                    y * window_size + x
                    for y in range(y_parity, window_size, 2)
                    for x in range(x_parity, window_size, 2)
                ]
            )
        self._positions = tuple(tuple(values) for values in positions)

    @property
    def sites_per_block(self) -> int:
        return len(self._positions[0])

    @property
    def positions(self) -> tuple[tuple[int, ...], ...]:
        return self._positions

    def _rank_to_group(
        self,
        windows_y: int,
        windows_x: int,
        latent_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        wy = torch.arange(windows_y, device=device)[:, None, None, None]
        wx = torch.arange(windows_x, device=device)[None, :, None, None]
        time = torch.arange(latent_frames, device=device)[None, None, :, None]
        group = torch.arange(self.num_groups, device=device)[None, None, None, :]
        group_y = group // 2
        group_x = group % 2
        phase_y = (wy + time) % 2
        phase_x = (wx + time // 2) % 2
        u_y = torch.bitwise_xor(group_y, phase_y)
        u_x = torch.bitwise_xor(group_x, phase_x)
        orientation = (wy + wx) % 2
        rank = torch.where(
            orientation == 0,
            2 * u_y + u_x,
            2 * u_x + u_y,
        )
        return rank.argsort(dim=-1)

    def to_blocks(self, grid: torch.Tensor) -> DynamicRankBlockTensor:
        if grid.ndim != 5:
            raise ValueError(f"Expected grid [B,T,H,W,C], got {tuple(grid.shape)}")
        batch, frames, height, width, channels = map(int, grid.shape)
        if height % self.window_size or width % self.window_size:
            raise ValueError(
                f"Latent size {(height, width)} must be divisible by "
                f"window_size={self.window_size}"
            )
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
        physical_groups = flat[:, :, :, :, positions, :]
        rank_to_group = self._rank_to_group(
            windows_y, windows_x, frames, grid.device
        )
        gather = rank_to_group[None, ..., None, None].expand(
            batch,
            windows_y,
            windows_x,
            frames,
            self.num_groups,
            self.sites_per_block,
            channels,
        )
        ranked = physical_groups.gather(4, gather)
        values = ranked.reshape(
            batch * windows_y * windows_x,
            frames * self.num_groups,
            self.sites_per_block,
            channels,
        )
        return DynamicRankBlockTensor(
            values,
            batch,
            frames,
            height,
            width,
            windows_y,
            windows_x,
            rank_to_group,
        )

    def from_blocks(self, blocks: DynamicRankBlockTensor) -> torch.Tensor:
        values = blocks.values
        channels = int(values.shape[-1])
        ranked = values.reshape(
            blocks.batch_size,
            blocks.windows_y,
            blocks.windows_x,
            blocks.latent_frames,
            self.num_groups,
            self.sites_per_block,
            channels,
        )
        gather = blocks.rank_to_group[None, ..., None, None].expand_as(ranked)
        physical_groups = torch.empty_like(ranked).scatter(4, gather, ranked)
        flat = torch.empty(
            blocks.batch_size,
            blocks.windows_y,
            blocks.windows_x,
            blocks.latent_frames,
            self.window_size**2,
            channels,
            dtype=values.dtype,
            device=values.device,
        )
        physical_sites = physical_groups.reshape(
            blocks.batch_size,
            blocks.windows_y,
            blocks.windows_x,
            blocks.latent_frames,
            self.window_size**2,
            channels,
        )
        positions = torch.tensor(
            self._positions, dtype=torch.long, device=values.device
        ).reshape(-1)
        scatter_positions = positions[None, None, None, None, :, None].expand_as(
            physical_sites
        )
        flat.scatter_(4, scatter_positions, physical_sites)
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

    def block_coordinates(
        self, blocks: DynamicRankBlockTensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        groups = blocks.rank_to_group.reshape(
            blocks.num_windows, blocks.latent_frames * self.num_groups
        )
        groups = groups.repeat(blocks.batch_size, 1)
        positions = torch.tensor(
            self._positions, dtype=torch.long, device=blocks.values.device
        )[groups]
        local_y = positions // self.window_size
        local_x = positions % self.window_size
        time = torch.arange(
            blocks.latent_frames, device=blocks.values.device
        ).repeat_interleave(self.num_groups)
        time = time[None, :, None].expand(
            blocks.batch_size * blocks.num_windows,
            -1,
            self.sites_per_block,
        )
        groups = groups[:, :, None].expand(-1, -1, self.sites_per_block)
        return time, groups, local_y, local_x
