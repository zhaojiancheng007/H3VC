from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .config_wan_vq import WanVQEntropyModelConfig
from .layout import BlockTensor, FourGroupBlockLayout
from .model import EntropyModelOutput
from .transformer import BlockCausalTransformer


@dataclass(frozen=True)
class WanVQPrediction:
    logits: tuple[torch.Tensor, ...]
    code_blocks: BlockTensor


class WanVQBinaryEntropyModel(nn.Module):
    """Four-group AR model for a frozen 4096x32 WanVQ tokenizer.

    The previous group supplies 32-D selected codebook vectors.  The current
    group's 4096-way index is represented as twelve binary classification
    targets, so the rate model exposes one binary head per bit.
    """

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
        self.output_head = nn.Linear(config.d_model, config.num_bits * 2)

    def _validate_features(self, quant_features: torch.Tensor) -> None:
        expected = self.config.codebook_dim
        if quant_features.ndim != 5 or quant_features.shape[1] != expected:
            raise ValueError(
                f"Expected quant_features [B,{expected},T,H,W], got {tuple(quant_features.shape)}"
            )
        if not quant_features.is_floating_point():
            raise TypeError("quant_features must be floating point")
        if not torch.isfinite(quant_features).all().item():
            raise FloatingPointError("quant_features contain non-finite values")

    def _validate_pair(
        self, quant_features: torch.Tensor, target_indices: torch.Tensor
    ) -> None:
        self._validate_features(quant_features)
        if target_indices.ndim != 4:
            raise ValueError(
                f"Expected target_indices [B,T,H,W], got {tuple(target_indices.shape)}"
            )
        if not target_indices.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("target_indices must be an integer tensor")
        if tuple(quant_features.shape[0:1] + quant_features.shape[2:]) != tuple(
            target_indices.shape
        ):
            raise ValueError(
                f"quant_features {tuple(quant_features.shape)} and target_indices "
                f"{tuple(target_indices.shape)} have incompatible shapes"
            )
        if target_indices.numel() and (
            int(target_indices.min()) < 0
            or int(target_indices.max()) >= self.config.codebook_size
        ):
            raise ValueError("target_indices contain values outside [0,4095]")

    def _position_embedding(
        self, latent_frames: int, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latent_frames > self.config.max_latent_frames:
            raise ValueError(
                f"latent_frames={latent_frames} exceeds max_latent_frames="
                f"{self.config.max_latent_frames}"
            )
        time, group, local_y, local_x = self.layout.coordinates(latent_frames, device=device)
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
        positions = torch.tensor(self.layout.positions, dtype=torch.long, device=device)
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
        if code_blocks.ndim != 4 or code_blocks.shape[-1] != self.config.codebook_dim:
            raise ValueError(
                f"Expected code blocks [B,S,M,{self.config.codebook_dim}], got "
                f"{tuple(code_blocks.shape)}"
            )
        batch_windows, stages, sites, _ = code_blocks.shape
        latent_frames, remainder = divmod(stages, self.layout.num_groups)
        if remainder:
            raise ValueError(f"Stage count {stages} is not divisible by four groups")
        projected = self.code_projection(code_blocks)
        inputs = torch.empty_like(projected)
        inputs[:, 0] = self.bos.expand(batch_windows, -1, -1)
        if stages > 1:
            inputs[:, 1:] = projected[:, :-1]
        target, source = self._position_embedding(
            latent_frames, device=code_blocks.device, dtype=projected.dtype
        )
        return inputs + target.unsqueeze(0) + source.unsqueeze(0)

    def _split_logits(self, logits: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(logits.reshape(*logits.shape[:-1], self.config.num_bits, 2).unbind(-2))

    def forward_blocks(self, code_blocks: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden = self.transformer(
            self._shifted_inputs(code_blocks), self.layout.sites_per_block
        )
        return self._split_logits(self.output_head(self.output_norm(hidden)))

    def predict_quant_features(self, quant_features: torch.Tensor) -> WanVQPrediction:
        self._validate_features(quant_features)
        code_grid = quant_features.permute(0, 2, 3, 4, 1).to(
            dtype=self.code_projection.weight.dtype
        )
        code_blocks = self.layout.to_blocks(code_grid)
        return WanVQPrediction(
            logits=self.forward_blocks(code_blocks.values), code_blocks=code_blocks
        )

    def forward(
        self, quant_features: torch.Tensor, target_indices: torch.Tensor
    ) -> EntropyModelOutput:
        self._validate_pair(quant_features, target_indices)
        prediction = self.predict_quant_features(quant_features)
        bits = target_indices.unsqueeze(-1).bitwise_right_shift(
            torch.arange(self.config.num_bits - 1, -1, -1, device=target_indices.device)
        ).bitwise_and(1)
        bit_blocks = self.layout.to_blocks(bits)
        losses: list[torch.Tensor] = []
        for bit, logits in enumerate(prediction.logits):
            targets = bit_blocks.values[..., bit]
            losses.append(
                F.cross_entropy(
                    logits.float().reshape(-1, 2), targets.reshape(-1), reduction="none"
                ).reshape(targets.shape) / math.log(2.0)
            )
        batch, frames, height, width = map(int, target_indices.shape)
        bits_per_digit = torch.stack([loss.sum() for loss in losses]) / (
            batch * frames * height * width
        )
        total_per_window = torch.stack(losses, dim=-1).sum(dim=(-1, -2, -3))
        estimated_bits = total_per_window.reshape(
            batch, prediction.code_blocks.num_windows
        ).sum(dim=1)
        bits_per_site = estimated_bits.sum() / (batch * frames * height * width)
        return EntropyModelOutput(
            logits=prediction.logits,
            estimated_bits=estimated_bits,
            bits_per_site=bits_per_site,
            bits_per_digit=bits_per_digit,
            latent_shape=(frames, height, width),
        )
