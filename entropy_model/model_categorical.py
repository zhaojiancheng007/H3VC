from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .config_categorical import CategoricalEntropyModelConfig
from .layout import BlockTensor, FourGroupBlockLayout
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
    """Four-group AR model predicting one 4096-way index at each position."""

    def __init__(self, config: CategoricalEntropyModelConfig) -> None:
        super().__init__()
        self.config = config
        self.layout = FourGroupBlockLayout(config.window_size)
        positions_per_group = self.layout.sites_per_block

        self.code_projection = nn.Linear(config.codebook_dim, config.d_model)
        self.bos = nn.Parameter(torch.zeros(1, positions_per_group, config.d_model))
        self.target_time = nn.Embedding(config.max_latent_frames, config.d_model)
        self.target_group = nn.Embedding(self.layout.num_groups, config.d_model)
        self.target_y = nn.Embedding(config.window_size, config.d_model)
        self.target_x = nn.Embedding(config.window_size, config.d_model)
        self.source_time = nn.Embedding(config.max_latent_frames, config.d_model)
        self.source_group = nn.Embedding(self.layout.num_groups, config.d_model)
        self.source_y = nn.Embedding(config.window_size, config.d_model)
        self.source_x = nn.Embedding(config.window_size, config.d_model)
        self.transformer = BlockCausalTransformer(
            config.d_model,
            config.num_layers,
            config.num_heads,
            config.ffn_dim,
        )
        self.output_norm = nn.RMSNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.codebook_size, bias=False)
        nn.init.zeros_(self.output_head.weight)

    def _validate(
        self, quantized: torch.Tensor, indices: torch.Tensor
    ) -> None:
        expected_grid = (
            quantized.shape[0],
            quantized.shape[2],
            quantized.shape[3],
            quantized.shape[4],
        )
        if quantized.ndim != 5 or quantized.shape[1] != self.config.codebook_dim:
            raise ValueError(
                f"Expected quantized [B,{self.config.codebook_dim},T,H,W], "
                f"got {tuple(quantized.shape)}"
            )
        if indices.dtype != torch.int64 or tuple(indices.shape) != expected_grid:
            raise ValueError(
                f"Expected int64 indices {expected_grid}, got "
                f"{indices.dtype} {tuple(indices.shape)}"
            )

    def _position_embeddings(
        self, latent_frames: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latent_frames > self.config.max_latent_frames:
            raise ValueError(
                f"latent_frames={latent_frames} exceeds "
                f"max_latent_frames={self.config.max_latent_frames}"
            )
        time, group, local_y, local_x = self.layout.coordinates(
            latent_frames, device=device
        )
        target = (
            self.target_time(time)
            + self.target_group(group)
            + self.target_y(local_y)
            + self.target_x(local_x)
        )

        previous = torch.arange(
            latent_frames * self.layout.num_groups, device=device
        ) - 1
        valid = previous >= 0
        previous = previous.clamp_min(0)
        source_time = previous // self.layout.num_groups
        source_group = previous % self.layout.num_groups
        positions = torch.tensor(
            self.layout.positions, dtype=torch.long, device=device
        )
        source_y = positions[source_group] // self.config.window_size
        source_x = positions[source_group] % self.config.window_size
        source = (
            self.source_time(source_time[:, None].expand_as(source_y))
            + self.source_group(source_group[:, None].expand_as(source_y))
            + self.source_y(source_y)
            + self.source_x(source_x)
        )
        source = source * valid[:, None, None]
        return target.to(dtype), source.to(dtype)

    def _shifted_inputs(self, code_blocks: torch.Tensor) -> torch.Tensor:
        projected = self.code_projection(code_blocks)
        inputs = torch.empty_like(projected)
        inputs[:, 0] = self.bos.expand(projected.shape[0], -1, -1)
        inputs[:, 1:] = projected[:, :-1]
        latent_frames, remainder = divmod(
            code_blocks.shape[1], self.layout.num_groups
        )
        if remainder:
            raise ValueError("Prediction groups are not divisible by four")
        target, source = self._position_embeddings(
            latent_frames, code_blocks.device, projected.dtype
        )
        return inputs + target.unsqueeze(0) + source.unsqueeze(0)

    def predict(self, quantized: torch.Tensor) -> tuple[torch.Tensor, BlockTensor]:
        grid = quantized.permute(0, 2, 3, 4, 1).to(
            self.code_projection.weight.dtype
        )
        blocks = self.layout.to_blocks(grid)
        hidden = self.transformer(
            self._shifted_inputs(blocks.values), self.layout.sites_per_block
        )
        return self.output_head(self.output_norm(hidden)), blocks

    def _targets(self, indices: torch.Tensor) -> torch.Tensor:
        return self.layout.to_blocks(indices.unsqueeze(-1)).values.squeeze(-1)

    def forward(
        self, quantized: torch.Tensor, indices: torch.Tensor
    ) -> CategoricalEntropyOutput:
        self._validate(quantized, indices)
        logits, blocks = self.predict(quantized)
        targets = self._targets(indices)
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
    ) -> SoftCategoricalEntropyOutput:
        self._validate(quantized, indices)
        expected = (*indices.shape, self.config.codebook_size)
        if tuple(target_probabilities.shape) != expected:
            raise ValueError(
                f"Expected target probabilities {expected}, "
                f"got {tuple(target_probabilities.shape)}"
            )
        logits, blocks = self.predict(quantized)
        targets = self._targets(indices)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            probability_blocks = self.layout.to_blocks(
                target_probabilities.float()
            ).values
            log2_probabilities = torch.log_softmax(logits.float(), dim=-1) / math.log(2.0)
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
