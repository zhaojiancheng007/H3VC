from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .config_row_group import RowGroupEntropyModelConfig
from .fsq import FSQSpec
from .layout_row_group import RowGroupLayout, RowGroupTensor
from .model import EntropyModelOutput
from .transformer import LayerKVCache
from .transformer_row_group import RowGroupBlockCausalTransformer


@dataclass(frozen=True)
class RowGroupPrediction:
    logits: tuple[torch.Tensor, ...]
    code_blocks: RowGroupTensor


class CosmosFSQRowGroupEntropyModel(nn.Module):
    """Four-row AR entropy model with one-frame plus decoded-row context."""

    def __init__(self, config: RowGroupEntropyModelConfig) -> None:
        super().__init__()
        self.config = config
        self.fsq = FSQSpec(config.levels)
        self.layout = RowGroupLayout(config.rows_per_group)

        self.code_projection = nn.Linear(self.fsq.dimensions, config.d_model)
        self.bos = nn.Parameter(torch.zeros(1, 1, config.d_model))

        self.target_time = nn.Embedding(config.max_latent_frames, config.d_model)
        self.target_y = nn.Embedding(config.max_latent_height, config.d_model)
        self.target_x = nn.Embedding(config.max_latent_width, config.d_model)
        self.source_time = nn.Embedding(config.max_latent_frames, config.d_model)
        self.source_y = nn.Embedding(config.max_latent_height, config.d_model)
        self.source_x = nn.Embedding(config.max_latent_width, config.d_model)

        self.transformer = RowGroupBlockCausalTransformer(
            config.d_model, config.num_layers, config.num_heads, config.ffn_dim
        )
        self.output_norm = nn.RMSNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, sum(config.levels))

    def _validate_shape(self, frames: int, height: int, width: int) -> None:
        if frames <= 0 or frames > self.config.max_latent_frames:
            raise ValueError(f"Unsupported latent frame count: {frames}")
        if height <= 0 or height > self.config.max_latent_height:
            raise ValueError(f"Unsupported latent height: {height}")
        if width <= 0 or width > self.config.max_latent_width:
            raise ValueError(f"Unsupported latent width: {width}")
        if height % self.config.rows_per_group:
            raise ValueError(
                f"Latent height {height} is not divisible by "
                f"rows_per_group={self.config.rows_per_group}"
            )

    def _validate_quant_features(self, quant_features: torch.Tensor) -> None:
        if quant_features.ndim != 5 or quant_features.shape[1] != self.fsq.dimensions:
            raise ValueError(
                "Expected quant_features [B,6,T,H,W], got "
                f"{tuple(quant_features.shape)}"
            )
        if not quant_features.is_floating_point():
            raise TypeError(f"quant_features must be floating point, got {quant_features.dtype}")
        _, _, frames, height, width = map(int, quant_features.shape)
        self._validate_shape(frames, height, width)
        if not torch.isfinite(quant_features).all().item():
            raise FloatingPointError("quant_features contain non-finite values")

    def _validate_feature_target_pair(
        self, quant_features: torch.Tensor, target_indices: torch.Tensor
    ) -> None:
        self._validate_quant_features(quant_features)
        if target_indices.ndim != 4:
            raise ValueError(f"Expected target_indices [B,T,H,W], got {tuple(target_indices.shape)}")
        expected = (target_indices.shape[0], self.fsq.dimensions, *target_indices.shape[1:])
        if tuple(quant_features.shape) != expected:
            raise ValueError(
                f"quant_features shape {tuple(quant_features.shape)} does not match "
                f"target_indices shape {tuple(target_indices.shape)}"
            )
        expected_features = self.fsq.indices_to_codes(
            target_indices, dtype=quant_features.dtype
        ).permute(0, 4, 1, 2, 3)
        if not torch.equal(quant_features, expected_features):
            error = float((quant_features.float() - expected_features.float()).abs().max().item())
            raise ValueError(
                "quant_features do not represent target_indices; "
                f"maximum absolute error is {error}"
            )

    def _stage_position(
        self,
        stage: int,
        *,
        latent_height: int,
        latent_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        groups = latent_height // self.config.rows_per_group
        frame, group = divmod(stage, groups)
        row_in_group = torch.arange(self.config.rows_per_group, device=device)
        x = torch.arange(latent_width, device=device)
        target_y = (group * self.config.rows_per_group + row_in_group[:, None]).expand(
            -1, latent_width
        ).reshape(-1)
        site_x = x[None, :].expand(self.config.rows_per_group, -1).reshape(-1)
        target = (
            self.target_time.weight[frame]
            + self.target_y(target_y)
            + self.target_x(site_x)
        )

        if stage == 0:
            source = torch.zeros_like(target)
        else:
            source_frame, source_group = divmod(stage - 1, groups)
            source_y = (
                source_group * self.config.rows_per_group + row_in_group[:, None]
            ).expand(-1, latent_width).reshape(-1)
            source = (
                self.source_time.weight[source_frame]
                + self.source_y(source_y)
                + self.source_x(site_x)
            )
        return target.to(dtype), source.to(dtype)

    @staticmethod
    def _cache_length(cache: tuple[LayerKVCache, ...]) -> int:
        lengths = {int(layer.key.shape[2]) for layer in cache}
        if len(lengths) != 1:
            raise RuntimeError(f"Transformer layers disagree on KV length: {sorted(lengths)}")
        return lengths.pop()

    def _trim_cache(
        self,
        cache: tuple[LayerKVCache, ...] | None,
        *,
        stage: int,
        groups_per_frame: int,
        sites: int,
    ) -> tuple[LayerKVCache, ...] | None:
        if stage == 0:
            if cache is not None:
                raise ValueError("stage 0 must not receive a KV cache")
            return None
        if cache is None:
            raise ValueError(f"stage {stage} requires a KV cache")

        previous_frame = (stage - 1) // groups_per_frame
        previous_start = 0 if previous_frame == 0 else (previous_frame - 1) * groups_per_frame + 1
        expected_stages = stage - previous_start
        actual_tokens = self._cache_length(cache)
        if actual_tokens != expected_stages * sites:
            raise ValueError(
                f"stage {stage} expected {expected_stages * sites} cached tokens, "
                f"got {actual_tokens}"
            )

        current_frame = stage // groups_per_frame
        current_start = 0 if current_frame == 0 else (current_frame - 1) * groups_per_frame + 1
        keep_stages = stage - current_start
        keep_tokens = keep_stages * sites
        if keep_tokens == actual_tokens:
            return cache
        if keep_tokens == 0:
            return tuple(
                LayerKVCache(key=layer.key[:, :, :0], value=layer.value[:, :, :0])
                for layer in cache
            )
        return tuple(
            LayerKVCache(
                key=layer.key[:, :, -keep_tokens:],
                value=layer.value[:, :, -keep_tokens:],
            )
            for layer in cache
        )

    def _split_logits(self, logits: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(logits.split(self.config.levels, dim=-1))

    def _all_stage_positions(
        self,
        *,
        latent_frames: int,
        latent_height: int,
        latent_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        groups = latent_height // self.config.rows_per_group
        stages = latent_frames * groups
        sites = self.config.rows_per_group * latent_width
        stage_ids = torch.arange(stages, device=device)
        target_frame = stage_ids // groups
        target_group = stage_ids % groups
        row_in_group = torch.arange(self.config.rows_per_group, device=device)
        site_x = torch.arange(latent_width, device=device).repeat(
            self.config.rows_per_group
        )
        target_y = (
            target_group[:, None, None] * self.config.rows_per_group
            + row_in_group[None, :, None]
        ).expand(-1, -1, latent_width).reshape(stages, sites)
        target = (
            self.target_time(target_frame)[:, None]
            + self.target_y(target_y)
            + self.target_x(site_x)[None]
        )

        source_stage = (stage_ids - 1).clamp_min(0)
        source_frame = source_stage // groups
        source_group = source_stage % groups
        source_y = (
            source_group[:, None, None] * self.config.rows_per_group
            + row_in_group[None, :, None]
        ).expand(-1, -1, latent_width).reshape(stages, sites)
        source = (
            self.source_time(source_frame)[:, None]
            + self.source_y(source_y)
            + self.source_x(site_x)[None]
        )
        source[0] = 0
        return target.to(dtype), source.to(dtype)

    def forward_step(
        self,
        previous_codes: torch.Tensor | None,
        *,
        stage: int,
        batch_size: int,
        latent_frames: int,
        latent_height: int,
        latent_width: int,
        cache: tuple[LayerKVCache, ...] | None,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[LayerKVCache, ...]]:
        self._validate_shape(latent_frames, latent_height, latent_width)
        groups = latent_height // self.config.rows_per_group
        stages = latent_frames * groups
        if stage < 0 or stage >= stages:
            raise ValueError(f"Invalid stage {stage} for {stages} stages")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        sites = self.config.rows_per_group * latent_width

        if stage == 0:
            if previous_codes is not None:
                raise ValueError("stage 0 must not receive previous_codes")
            current = self.bos.expand(batch_size, sites, -1)
        else:
            expected = (batch_size, sites, self.fsq.dimensions)
            if previous_codes is None or previous_codes.ndim != 3:
                actual = None if previous_codes is None else tuple(previous_codes.shape)
                raise ValueError(f"stage {stage} expected previous_codes [B,{sites},6], got {actual}")
            if tuple(previous_codes.shape) != expected:
                raise ValueError(
                    f"stage {stage} expected previous_codes {expected}, "
                    f"got {tuple(previous_codes.shape)}"
                )
            current = self.code_projection(previous_codes.contiguous())

        target, source = self._stage_position(
            stage,
            latent_height=latent_height,
            latent_width=latent_width,
            device=current.device,
            dtype=current.dtype,
        )
        current = current + target.unsqueeze(0) + source.unsqueeze(0)
        cache = self._trim_cache(
            cache,
            stage=stage,
            groups_per_frame=groups,
            sites=sites,
        )
        if current.device.type == "cuda":
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                hidden, cache = self.transformer.forward_step(current, cache)
        else:
            hidden, cache = self.transformer.forward_step(current, cache)
        logits = self._split_logits(self.output_head(self.output_norm(hidden)))
        return logits, cache

    def predict_quant_features(self, quant_features: torch.Tensor) -> RowGroupPrediction:
        self._validate_quant_features(quant_features)
        batch, _, frames, height, width = map(int, quant_features.shape)
        code_grid = quant_features.permute(0, 2, 3, 4, 1).to(
            dtype=self.code_projection.weight.dtype
        )
        code_blocks = self.layout.to_blocks(code_grid)
        projected = self.code_projection(code_blocks.values.contiguous())
        bos = self.bos.to(projected.dtype).expand(batch, 1, projected.shape[2], -1)
        shifted = torch.cat((bos, projected[:, :-1]), dim=1)
        target, source = self._all_stage_positions(
            latent_frames=frames,
            latent_height=height,
            latent_width=width,
            device=shifted.device,
            dtype=shifted.dtype,
        )
        hidden = self.transformer(
            shifted + target.unsqueeze(0) + source.unsqueeze(0),
            groups_per_frame=height // self.config.rows_per_group,
            sites_per_group=self.config.rows_per_group * width,
        )
        logits = self._split_logits(self.output_head(self.output_norm(hidden)))
        return RowGroupPrediction(
            logits=logits,
            code_blocks=code_blocks,
        )

    def forward(
        self, quant_features: torch.Tensor, target_indices: torch.Tensor
    ) -> EntropyModelOutput:
        self._validate_feature_target_pair(quant_features, target_indices)
        prediction = self.predict_quant_features(quant_features)
        digit_blocks = self.layout.to_blocks(self.fsq.indices_to_digits(target_indices))

        bit_tensors: list[torch.Tensor] = []
        for dimension, logits in enumerate(prediction.logits):
            targets = digit_blocks.values[..., dimension]
            bits = F.cross_entropy(
                logits.float().reshape(-1, self.config.levels[dimension]),
                targets.reshape(-1),
                reduction="none",
            ).reshape(targets.shape) / math.log(2.0)
            bit_tensors.append(bits)

        batch, frames, height, width = map(int, target_indices.shape)
        bits_per_digit = torch.stack([bits.sum() for bits in bit_tensors]) / (
            batch * frames * height * width
        )
        estimated_bits = torch.stack(bit_tensors, dim=-1).sum(dim=(1, 2, 3))
        bits_per_site = estimated_bits.sum() / (batch * frames * height * width)
        return EntropyModelOutput(
            logits=prediction.logits,
            estimated_bits=estimated_bits,
            bits_per_site=bits_per_site,
            bits_per_digit=bits_per_digit,
            latent_shape=(frames, height, width),
        )
