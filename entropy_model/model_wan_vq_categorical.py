from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .config_wan_vq import WanVQEntropyModelConfig
from .layout import BlockTensor, FourGroupBlockLayout
from .transformer import BlockCausalTransformer


@dataclass(frozen=True)
class WanVQCategoricalOutput:
    logits: torch.Tensor
    estimated_bits: torch.Tensor
    bits_per_site: torch.Tensor
    latent_shape: tuple[int, int, int]


class WanVQCategoricalEntropyModel(nn.Module):
    """Four-group AR model that predicts one 4096-way VQ index per position."""

    def __init__(self, config: WanVQEntropyModelConfig) -> None:
        super().__init__()
        self.config = config
        self.layout = FourGroupBlockLayout(config.window_size)
        sites = self.layout.sites_per_block

        self.code_projection = nn.Linear(config.codebook_dim, config.d_model)
        self.bos = nn.Parameter(torch.zeros(1, sites, config.d_model))
        self.target_time = nn.Embedding(config.max_latent_frames, config.d_model)
        self.target_group = nn.Embedding(self.layout.num_groups, config.d_model)
        self.target_y = nn.Embedding(config.window_size, config.d_model)
        self.target_x = nn.Embedding(config.window_size, config.d_model)
        self.source_time = nn.Embedding(config.max_latent_frames, config.d_model)
        self.source_group = nn.Embedding(self.layout.num_groups, config.d_model)
        self.source_y = nn.Embedding(config.window_size, config.d_model)
        self.source_x = nn.Embedding(config.window_size, config.d_model)

        self.transformer = BlockCausalTransformer(
            config.d_model, config.num_layers, config.num_heads, config.ffn_dim
        )
        self.output_norm = nn.RMSNorm(config.d_model)
        self.output_head = nn.Linear(
            config.d_model, config.codebook_size, bias=False
        )
        nn.init.zeros_(self.output_head.weight)

    def _validate_pair(
        self, quant_features: torch.Tensor, target_indices: torch.Tensor
    ) -> None:
        if (
            quant_features.ndim != 5
            or quant_features.shape[1] != self.config.codebook_dim
        ):
            raise ValueError(
                f"Expected quant_features [B,{self.config.codebook_dim},T,H,W], "
                f"got {tuple(quant_features.shape)}"
            )
        if target_indices.ndim != 4:
            raise ValueError(
                f"Expected target_indices [B,T,H,W], got {tuple(target_indices.shape)}"
            )
        feature_grid_shape = (
            quant_features.shape[0],
            quant_features.shape[2],
            quant_features.shape[3],
            quant_features.shape[4],
        )
        if feature_grid_shape != tuple(target_indices.shape):
            raise ValueError(
                f"Feature grid {feature_grid_shape} does not match index grid "
                f"{tuple(target_indices.shape)}"
            )
        if not quant_features.is_floating_point():
            raise TypeError("quant_features must be floating point")
        if target_indices.dtype != torch.int64:
            raise TypeError("target_indices must be torch.int64")
        if int(target_indices.min()) < 0 or int(target_indices.max()) >= self.config.codebook_size:
            raise ValueError("target_indices contain values outside [0,4095]")

    def _position_embeddings(
        self, latent_frames: int, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latent_frames > self.config.max_latent_frames:
            raise ValueError(
                f"latent_frames={latent_frames} exceeds max_latent_frames="
                f"{self.config.max_latent_frames}"
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

        source_stage = torch.arange(
            latent_frames * self.layout.num_groups, device=device
        ) - 1
        source_valid = source_stage >= 0
        source_stage = source_stage.clamp_min(0)
        source_time = source_stage // self.layout.num_groups
        source_group = source_stage % self.layout.num_groups
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
        source = source * source_valid[:, None, None]
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
            raise ValueError("The number of prediction groups must be divisible by four")
        target, source = self._position_embeddings(
            latent_frames, device=code_blocks.device, dtype=projected.dtype
        )
        return inputs + target.unsqueeze(0) + source.unsqueeze(0)

    def predict_quant_features(
        self, quant_features: torch.Tensor
    ) -> tuple[torch.Tensor, BlockTensor]:
        code_grid = quant_features.permute(0, 2, 3, 4, 1).to(
            self.code_projection.weight.dtype
        )
        code_blocks = self.layout.to_blocks(code_grid)
        hidden = self.transformer(
            self._shifted_inputs(code_blocks.values), self.layout.sites_per_block
        )
        logits = self.output_head(self.output_norm(hidden))
        return logits, code_blocks

    def forward(
        self, quant_features: torch.Tensor, target_indices: torch.Tensor
    ) -> WanVQCategoricalOutput:
        self._validate_pair(quant_features, target_indices)
        logits, code_blocks = self.predict_quant_features(quant_features)
        target_grid = target_indices.unsqueeze(-1)
        targets = self.layout.to_blocks(target_grid).values.squeeze(-1)
        bits = F.cross_entropy(
            logits.float().reshape(-1, self.config.codebook_size),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape) / math.log(2.0)

        batch, frames, height, width = map(int, target_indices.shape)
        bits_per_window = bits.sum(dim=(-1, -2))
        estimated_bits = bits_per_window.reshape(
            batch, code_blocks.num_windows
        ).sum(dim=1)
        bits_per_site = estimated_bits.sum() / (batch * frames * height * width)
        return WanVQCategoricalOutput(
            logits=logits,
            estimated_bits=estimated_bits,
            bits_per_site=bits_per_site,
            latent_shape=(frames, height, width),
        )
