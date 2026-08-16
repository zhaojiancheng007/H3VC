from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import EntropyModelConfig
from .fsq import FSQSpec
from .layout import BlockTensor, FourGroupBlockLayout
from .transformer import BlockCausalTransformer, LayerKVCache


@dataclass(frozen=True)
class EntropyModelOutput:
    logits: tuple[torch.Tensor, ...]
    estimated_bits: torch.Tensor
    bits_per_site: torch.Tensor
    bits_per_digit: torch.Tensor
    latent_shape: tuple[int, int, int]


class CosmosFSQEntropyModel(nn.Module):
    def __init__(self, config: EntropyModelConfig) -> None:
        super().__init__()
        self.config = config
        self.fsq = FSQSpec(config.levels)
        self.layout = FourGroupBlockLayout(config.window_size)
        sites = self.layout.sites_per_block

        self.code_projection = nn.Linear(self.fsq.dimensions, config.d_model)
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
        self.output_head = nn.Linear(config.d_model, sum(config.levels))

    def _position_embedding(
        self, latent_frames: int, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latent_frames > self.config.max_latent_frames:
            raise ValueError(
                f"latent_frames={latent_frames} exceeds max_latent_frames={self.config.max_latent_frames}"
            )
        time, group, local_y, local_x = self.layout.coordinates(latent_frames, device=device)
        target = (
            self.target_time(time)
            + self.target_group(group)
            + self.target_y(local_y)
            + self.target_x(local_x)
        )

        source_stage = torch.arange(latent_frames * self.layout.num_groups, device=device) - 1
        source_valid = source_stage >= 0
        source_stage = source_stage.clamp_min(0)
        source_time_index = source_stage // self.layout.num_groups
        source_group_index = source_stage % self.layout.num_groups
        positions = torch.tensor(self.layout.positions, dtype=torch.long, device=device)
        source_y = positions[source_group_index] // self.config.window_size
        source_x = positions[source_group_index] % self.config.window_size
        source = (
            self.source_time(source_time_index[:, None].expand_as(source_y))
            + self.source_group(source_group_index[:, None].expand_as(source_y))
            + self.source_y(source_y)
            + self.source_x(source_x)
        )
        source = source * source_valid[:, None, None]
        return target.to(dtype), source.to(dtype)

    def _shifted_inputs(self, code_blocks: torch.Tensor) -> torch.Tensor:
        if code_blocks.ndim != 4 or code_blocks.shape[-1] != self.fsq.dimensions:
            raise ValueError(
                f"Expected code blocks [B,S,M,{self.fsq.dimensions}], got {tuple(code_blocks.shape)}"
            )
        batch_windows, stages, sites, _ = code_blocks.shape
        latent_frames, remainder = divmod(stages, self.layout.num_groups)
        if remainder:
            raise ValueError(f"Stage count {stages} is not divisible by {self.layout.num_groups}")
        projected = self.code_projection(code_blocks)
        inputs = torch.empty_like(projected)
        inputs[:, 0] = self.bos.expand(batch_windows, -1, -1)
        if stages > 1:
            inputs[:, 1:] = projected[:, :-1]
        target, source = self._position_embedding(
            latent_frames, device=code_blocks.device, dtype=projected.dtype
        )
        if target.shape != (stages, sites, self.config.d_model):
            raise RuntimeError(
                f"Position shape {tuple(target.shape)} does not match {(stages, sites, self.config.d_model)}"
            )
        return inputs + target.unsqueeze(0) + source.unsqueeze(0)

    def _split_logits(self, logits: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(logits.split(self.config.levels, dim=-1))

    def forward_blocks(self, code_blocks: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden = self.transformer(
            self._shifted_inputs(code_blocks), self.layout.sites_per_block
        )
        return self._split_logits(self.output_head(self.output_norm(hidden)))

    def forward(self, indices: torch.Tensor) -> EntropyModelOutput:
        if indices.ndim != 4:
            raise ValueError(f"Expected indices [B,T,H,W], got {tuple(indices.shape)}")
        batch, frames, height, width = map(int, indices.shape)
        digits = self.fsq.indices_to_digits(indices)
        codes = self.fsq.digits_to_codes(digits, dtype=self.code_projection.weight.dtype)
        digit_blocks = self.layout.to_blocks(digits)
        code_blocks = self.layout.to_blocks(codes)
        logits = self.forward_blocks(code_blocks.values)

        bit_tensors: list[torch.Tensor] = []
        for dimension, dimension_logits in enumerate(logits):
            targets = digit_blocks.values[..., dimension]
            bits = F.cross_entropy(
                dimension_logits.float().reshape(-1, self.config.levels[dimension]),
                targets.reshape(-1),
                reduction="none",
            ).reshape(targets.shape) / math.log(2.0)
            bit_tensors.append(bits)

        bits_per_digit = torch.stack([bits.sum() for bits in bit_tensors]) / (
            batch * frames * height * width
        )
        total_per_window = torch.stack(bit_tensors, dim=-1).sum(dim=(-1, -2, -3))
        estimated_bits = total_per_window.reshape(batch, code_blocks.num_windows).sum(dim=1)
        bits_per_site = estimated_bits.sum() / (batch * frames * height * width)
        return EntropyModelOutput(
            logits=logits,
            estimated_bits=estimated_bits,
            bits_per_site=bits_per_site,
            bits_per_digit=bits_per_digit,
            latent_shape=(frames, height, width),
        )

    def forward_step(
        self,
        previous_codes: torch.Tensor | None,
        *,
        stage: int,
        batch_windows: int,
        cache: tuple[LayerKVCache | None, ...] | None,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[LayerKVCache, ...]]:
        if stage < 0 or stage >= self.config.max_latent_frames * self.layout.num_groups:
            raise ValueError(f"Invalid stage {stage}")
        sites = self.layout.sites_per_block
        if stage == 0:
            if previous_codes is not None:
                raise ValueError("stage 0 must not receive previous_codes")
            current = self.bos.expand(batch_windows, -1, -1)
        else:
            expected = (batch_windows, sites, self.fsq.dimensions)
            if previous_codes is None or tuple(previous_codes.shape) != expected:
                actual = None if previous_codes is None else tuple(previous_codes.shape)
                raise ValueError(f"stage {stage} expected previous_codes {expected}, got {actual}")
            current = self.code_projection(previous_codes.contiguous())

        latent_frames = stage // self.layout.num_groups + 1
        target, source = self._position_embedding(
            latent_frames, device=current.device, dtype=current.dtype
        )
        current = current + target[stage].unsqueeze(0) + source[stage].unsqueeze(0)
        hidden, cache = self.transformer.forward_step(current, cache)
        logits = self._split_logits(self.output_head(self.output_norm(hidden)))
        return logits, cache
