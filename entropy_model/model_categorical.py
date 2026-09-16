from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .config_categorical import CategoricalEntropyModelConfig
from .layout import BlockTensor, SpatiotemporalWindowLayout
from .transformer import BlockCausalTransformer


@dataclass(frozen=True)
class CategoricalEntropyOutput:
    logits: torch.Tensor
    estimated_bits: torch.Tensor
    bits_per_index: torch.Tensor


@dataclass(frozen=True)
class SoftCategoricalEntropyOutput:
    soft_estimated_bits: torch.Tensor
    hard_estimated_bits: torch.Tensor
    soft_bits_per_index: torch.Tensor
    hard_bits_per_index: torch.Tensor


class CategoricalEntropyModel(nn.Module):
    """One shared 3D-window entropy model with an explicit temporal length K."""

    def __init__(self, config: CategoricalEntropyModelConfig) -> None:
        super().__init__()
        self.config = config
        sites = config.window_size**2 // 4
        self.code_projection = nn.Linear(config.codebook_dim, config.d_model)
        self.bos = nn.Parameter(torch.zeros(1, sites, config.d_model))
        self.target_time = nn.Embedding(config.max_latent_frames, config.d_model)
        self.target_group = nn.Embedding(4, config.d_model)
        self.target_y = nn.Embedding(config.window_size, config.d_model)
        self.target_x = nn.Embedding(config.window_size, config.d_model)
        self.source_time = nn.Embedding(config.max_latent_frames, config.d_model)
        self.source_group = nn.Embedding(4, config.d_model)
        self.source_y = nn.Embedding(config.window_size, config.d_model)
        self.source_x = nn.Embedding(config.window_size, config.d_model)
        self.transformer = BlockCausalTransformer(
            config.d_model,
            config.num_layers,
            config.num_heads,
            config.ffn_dim,
        )
        self.output_norm = nn.RMSNorm(config.d_model)
        self.output_head = nn.Linear(
            config.d_model, config.codebook_size, bias=False
        )
        nn.init.zeros_(self.output_head.weight)

    def layout(self, k: int) -> SpatiotemporalWindowLayout:
        return SpatiotemporalWindowLayout(self.config.window_size, k)

    def _validate(self, quantized: torch.Tensor, indices: torch.Tensor) -> None:
        if quantized.ndim != 5 or quantized.shape[1] != self.config.codebook_dim:
            raise ValueError(
                f"Expected quantized [B,{self.config.codebook_dim},T,H,W], "
                f"got {tuple(quantized.shape)}"
            )
        expected = (
            quantized.shape[0],
            quantized.shape[2],
            quantized.shape[3],
            quantized.shape[4],
        )
        if indices.dtype != torch.int64 or tuple(indices.shape) != expected:
            raise ValueError(
                f"Expected int64 indices {expected}, got "
                f"{indices.dtype} {tuple(indices.shape)}"
            )
        if quantized.shape[2] > self.config.max_latent_frames:
            raise ValueError(
                f"latent_frames={quantized.shape[2]} exceeds "
                f"max_latent_frames={self.config.max_latent_frames}"
            )

    def _position_embeddings(
        self,
        blocks: BlockTensor,
        layout: SpatiotemporalWindowLayout,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time, group, local_y, local_x = layout.coordinates(
            blocks.latent_frames, device=self.bos.device
        )
        target = (
            self.target_time(time)
            + self.target_group(group)
            + self.target_y(local_y)
            + self.target_x(local_x)
        )
        source_time = time.roll(1, dims=0)
        source_group = group.roll(1, dims=0)
        source_y = local_y.roll(1, dims=0)
        source_x = local_x.roll(1, dims=0)
        source = (
            self.source_time(source_time)
            + self.source_group(source_group)
            + self.source_y(source_y)
            + self.source_x(source_x)
        )
        source[0] = 0
        return target.to(dtype), source.to(dtype)

    def _shifted_inputs(
        self, blocks: BlockTensor, layout: SpatiotemporalWindowLayout
    ) -> torch.Tensor:
        projected = self.code_projection(blocks.values)
        inputs = torch.empty_like(projected)
        inputs[:, 0] = self.bos.expand(projected.shape[0], -1, -1)
        inputs[:, 1:] = projected[:, :-1]
        target, source = self._position_embeddings(blocks, layout, projected.dtype)
        return inputs + target.unsqueeze(0) + source.unsqueeze(0)

    def step_inputs(
        self,
        blocks: BlockTensor,
        rank: int,
        previous_vectors: torch.Tensor | None,
    ) -> torch.Tensor:
        layout = self.layout(blocks.temporal_window_size)
        if rank < 0 or rank >= blocks.ranks_per_window:
            raise ValueError(
                f"rank must be in [0,{blocks.ranks_per_window}), got {rank}"
            )
        if rank == 0:
            if previous_vectors is not None:
                raise ValueError("rank zero must use BOS, not previous vectors")
            inputs = self.bos.expand(blocks.values.shape[0], -1, -1)
        else:
            if previous_vectors is None:
                raise ValueError("nonzero rank requires previous vectors")
            inputs = self.code_projection(previous_vectors)
        target, source = self._position_embeddings(blocks, layout, inputs.dtype)
        return inputs + target[rank] + source[rank]

    def predict(
        self, quantized: torch.Tensor, *, k: int
    ) -> tuple[torch.Tensor, BlockTensor]:
        layout = self.layout(k)
        grid = quantized.permute(0, 2, 3, 4, 1).to(
            self.code_projection.weight.dtype
        )
        blocks = layout.to_blocks(grid)
        hidden = self.transformer(
            self._shifted_inputs(blocks, layout), layout.sites_per_block
        )
        return self.output_head(self.output_norm(hidden)), blocks

    def _targets(self, indices: torch.Tensor, *, k: int) -> torch.Tensor:
        return self.layout(k).to_blocks(indices.unsqueeze(-1)).values.squeeze(-1)

    def forward(
        self, quantized: torch.Tensor, indices: torch.Tensor, *, k: int
    ) -> CategoricalEntropyOutput:
        self._validate(quantized, indices)
        logits, blocks = self.predict(quantized, k=k)
        targets = self._targets(indices, k=k)
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

    def forward_soft(
        self,
        quantized: torch.Tensor,
        indices: torch.Tensor,
        target_probabilities: torch.Tensor,
        *,
        k: int,
    ) -> SoftCategoricalEntropyOutput:
        self._validate(quantized, indices)
        expected = (*indices.shape, self.config.codebook_size)
        if tuple(target_probabilities.shape) != expected:
            raise ValueError(
                f"Expected target probabilities {expected}, "
                f"got {tuple(target_probabilities.shape)}"
            )
        logits, blocks = self.predict(quantized, k=k)
        targets = self._targets(indices, k=k)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            probability_blocks = self.layout(k).to_blocks(
                target_probabilities.float()
            ).values
            log2_probabilities = (
                torch.log_softmax(logits.float(), dim=-1) / math.log(2.0)
            )
            soft_bits = -(probability_blocks * log2_probabilities).sum(dim=-1)
            hard_bits = -log2_probabilities.gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)
            soft_estimated_bits = soft_bits.sum(dim=(-1, -2)).reshape(
                indices.shape[0], blocks.num_windows
            ).sum(dim=1)
            hard_estimated_bits = hard_bits.sum(dim=(-1, -2)).reshape(
                indices.shape[0], blocks.num_windows
            ).sum(dim=1)
        return SoftCategoricalEntropyOutput(
            soft_estimated_bits=soft_estimated_bits,
            hard_estimated_bits=hard_estimated_bits,
            soft_bits_per_index=soft_estimated_bits.sum() / indices.numel(),
            hard_bits_per_index=hard_estimated_bits.sum() / indices.numel(),
        )


__all__ = [
    "CategoricalEntropyOutput",
    "SoftCategoricalEntropyOutput",
    "CategoricalEntropyModel",
]
