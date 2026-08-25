from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F

from .layout_dynamic_rank import DynamicRankBlockTensor
from .model_categorical import CategoricalEntropyOutput
from .model_categorical_dynamic_rank_reliable import (
    CategoricalEntropyModelDynamicRankReliable,
)
from .transformer import LayerKVCache, TransformerBlock


def sinusoidal_window_positions(
    windows_y: int,
    windows_x: int,
    channels: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if channels % 4:
        raise ValueError(f"channels must be divisible by four, got {channels}")
    half = channels // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(0, half, 2, device=device, dtype=torch.float32)
        / half
    )

    def encode(coordinates: torch.Tensor) -> torch.Tensor:
        angles = coordinates.float()[:, None] * frequencies[None]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)

    y = torch.arange(windows_y, device=device).repeat_interleave(windows_x)
    x = torch.arange(windows_x, device=device).repeat(windows_y)
    return torch.cat((encode(y), encode(x)), dim=-1).to(dtype)


class TemporalCrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query_norm = nn.RMSNorm(d_model)
        self.memory_norm = nn.RMSNorm(d_model)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.output.weight)

    def _heads(self, values: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
        batch, length, channels = values.shape
        return projection(values).reshape(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        query = self._heads(self.query_norm(queries), self.query)
        normalized_memory = self.memory_norm(memory)
        key = self._heads(normalized_memory, self.key)
        value = self._heads(normalized_memory, self.value)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape_as(queries)
        return self.output(attended)


class TemporalWindowContext(nn.Module):
    """Factorized all-window history with strict frame causality."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.spatial_mixer = TransformerBlock(d_model, num_heads, ffn_dim)
        self.temporal_bos = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cross_attention = TemporalCrossAttention(d_model, num_heads)

    @staticmethod
    def _window_group_ids(
        blocks: DynamicRankBlockTensor,
    ) -> torch.Tensor:
        return blocks.rank_to_group.reshape(
            blocks.num_windows, blocks.latent_frames, blocks.rank_to_group.shape[-1]
        )

    def _contextualize_frames(
        self,
        projected_blocks: torch.Tensor,
        blocks: DynamicRankBlockTensor,
        source_time: nn.Embedding,
        source_group: nn.Embedding,
    ) -> torch.Tensor:
        batch = blocks.batch_size
        windows = blocks.num_windows
        frames = blocks.latent_frames
        groups = int(blocks.rank_to_group.shape[-1])
        channels = int(projected_blocks.shape[-1])
        summaries = projected_blocks.reshape(
            batch, windows, frames, groups, projected_blocks.shape[-2], channels
        ).mean(dim=-2)
        summaries = summaries.permute(0, 2, 1, 3, 4)

        group_ids = self._window_group_ids(blocks).permute(1, 0, 2)
        group_ids = group_ids[None].expand(batch, -1, -1, -1)
        time_ids = torch.arange(frames, device=projected_blocks.device)
        time_embedding = source_time(time_ids)[:, None, None]
        window_embedding = sinusoidal_window_positions(
            blocks.windows_y,
            blocks.windows_x,
            channels,
            device=projected_blocks.device,
            dtype=projected_blocks.dtype,
        )[None, :, None]
        summaries = (
            summaries
            + time_embedding[None].to(projected_blocks.dtype)
            + source_group(group_ids).to(projected_blocks.dtype)
            + window_embedding[None]
        )

        flat = summaries.reshape(batch * frames, windows * groups, channels)
        spatial_mask = torch.zeros(
            1,
            1,
            windows * groups,
            windows * groups,
            device=flat.device,
            dtype=flat.dtype,
        )
        mixed = self.spatial_mixer(flat, spatial_mask)
        return mixed.reshape(batch, frames, windows, groups, channels)

    @staticmethod
    def _temporal_attention_mask(
        frames: int,
        groups: int,
        sites: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        target_time = torch.arange(frames, device=device).repeat_interleave(
            groups * sites
        )
        source_time = torch.arange(frames, device=device).repeat_interleave(groups)
        allowed_history = source_time[None] < target_time[:, None]
        allowed = torch.cat(
            (
                torch.ones(target_time.numel(), 1, device=device, dtype=torch.bool),
                allowed_history,
            ),
            dim=1,
        )
        mask = torch.zeros(allowed.shape, device=device, dtype=dtype)
        mask.masked_fill_(~allowed, float("-inf"))
        return mask[None, None]

    def forward(
        self,
        local_hidden: torch.Tensor,
        projected_blocks: torch.Tensor,
        blocks: DynamicRankBlockTensor,
        source_time: nn.Embedding,
        source_group: nn.Embedding,
    ) -> torch.Tensor:
        batch_windows, num_blocks, sites, channels = local_hidden.shape
        groups = int(blocks.rank_to_group.shape[-1])
        mixed = self._contextualize_frames(
            projected_blocks, blocks, source_time, source_group
        )
        memory = mixed.permute(0, 2, 1, 3, 4).reshape(
            batch_windows, blocks.latent_frames * groups, channels
        )
        bos = self.temporal_bos.to(memory.dtype).expand(batch_windows, -1, -1)
        memory = torch.cat((bos, memory), dim=1)
        queries = local_hidden.reshape(batch_windows, num_blocks * sites, channels)
        mask = self._temporal_attention_mask(
            blocks.latent_frames,
            groups,
            sites,
            device=queries.device,
            dtype=queries.dtype,
        )
        return self.cross_attention(queries, memory, mask).reshape_as(local_hidden)

    def empty_memory(
        self, batch_windows: int, *, dtype: torch.dtype
    ) -> torch.Tensor:
        return self.temporal_bos.to(dtype).expand(batch_windows, -1, -1)

    def contextualize_completed_frame(
        self,
        projected_frame: torch.Tensor,
        blocks: DynamicRankBlockTensor,
        frame_index: int,
        source_time: nn.Embedding,
        source_group: nn.Embedding,
    ) -> torch.Tensor:
        batch = blocks.batch_size
        windows = blocks.num_windows
        groups = int(blocks.rank_to_group.shape[-1])
        channels = int(projected_frame.shape[-1])
        if (
            projected_frame.ndim != 4
            or projected_frame.shape[0] != batch * windows
            or projected_frame.shape[1] != groups
        ):
            raise ValueError(
                f"Unexpected completed-frame blocks: {tuple(projected_frame.shape)}"
            )
        summaries = projected_frame.reshape(
            batch, windows, groups, projected_frame.shape[-2], channels
        ).mean(dim=-2)
        group_ids = self._window_group_ids(blocks)[:, frame_index]
        group_ids = group_ids[None].expand(batch, -1, -1)
        window_embedding = sinusoidal_window_positions(
            blocks.windows_y,
            blocks.windows_x,
            channels,
            device=projected_frame.device,
            dtype=projected_frame.dtype,
        )[None, :, None]
        summaries = (
            summaries
            + source_time.weight[frame_index].to(projected_frame.dtype)
            + source_group(group_ids).to(projected_frame.dtype)
            + window_embedding
        )
        flat = summaries.reshape(batch, windows * groups, channels)
        spatial_mask = torch.zeros(
            1,
            1,
            windows * groups,
            windows * groups,
            device=flat.device,
            dtype=flat.dtype,
        )
        mixed = self.spatial_mixer(flat, spatial_mask)
        return mixed.reshape(batch, windows, groups, channels).reshape(
            batch * windows, groups, channels
        )

    def attend_step(
        self, local_hidden: torch.Tensor, temporal_memory: torch.Tensor
    ) -> torch.Tensor:
        return self.cross_attention(local_hidden, temporal_memory, None)


class CategoricalEntropyModelTemporalWindow(nn.Module):
    """Clean dynamic-rank entropy model with all-window historical context."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.local = CategoricalEntropyModelDynamicRankReliable(config)
        self.temporal = TemporalWindowContext(
            config.d_model, config.num_heads, config.ffn_dim
        )

    @property
    def layout(self):
        return self.local.layout

    def load_parent_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        self.local.load_state_dict(state_dict, strict=True)

    def predict(
        self, quantized: torch.Tensor
    ) -> tuple[torch.Tensor, DynamicRankBlockTensor]:
        grid = quantized.permute(0, 2, 3, 4, 1).to(
            self.local.code_projection.weight.dtype
        )
        blocks = self.layout.to_blocks(grid)
        projected = self.local.code_projection(blocks.values)
        local_hidden = self.local.transformer(
            self.local._shifted_dynamic_inputs(blocks), self.layout.sites_per_block
        )
        temporal_delta = self.temporal(
            local_hidden,
            projected,
            blocks,
            self.local.source_time,
            self.local.source_group,
        )
        hidden = local_hidden + temporal_delta
        return self.local.output_head(self.local.output_norm(hidden)), blocks

    def forward(
        self, quantized: torch.Tensor, indices: torch.Tensor
    ) -> CategoricalEntropyOutput:
        self.local._validate(quantized, indices)
        logits, blocks = self.predict(quantized)
        targets = self.local._targets(indices)
        bits = F.cross_entropy(
            logits.float().reshape(-1, self.config.codebook_size),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape) / math.log(2.0)
        estimated_bits = bits.sum(dim=(-1, -2)).reshape(
            indices.shape[0], blocks.num_windows
        ).sum(dim=1)
        return CategoricalEntropyOutput(
            logits=logits,
            estimated_bits=estimated_bits,
            bits_per_index=estimated_bits.sum() / indices.numel(),
        )

    def forward_block(
        self,
        inputs: torch.Tensor,
        local_cache: tuple[LayerKVCache | None, ...] | None,
        temporal_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        local_hidden, updated_cache = self.local.transformer.forward_step(
            inputs, local_cache
        )
        hidden = local_hidden + self.temporal.attend_step(
            local_hidden, temporal_memory
        )
        logits = self.local.output_head(self.local.output_norm(hidden))
        return logits, updated_cache

    def empty_temporal_memory(
        self, batch_windows: int, *, dtype: torch.dtype
    ) -> torch.Tensor:
        return self.temporal.empty_memory(batch_windows, dtype=dtype)

    def append_completed_frame(
        self,
        temporal_memory: torch.Tensor,
        completed_vectors: torch.Tensor,
        blocks: DynamicRankBlockTensor,
        frame_index: int,
    ) -> torch.Tensor:
        projected = self.local.code_projection(completed_vectors)
        summaries = self.temporal.contextualize_completed_frame(
            projected,
            blocks,
            frame_index,
            self.local.source_time,
            self.local.source_group,
        )
        return torch.cat((temporal_memory, summaries), dim=1)
