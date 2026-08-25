from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .layout_dynamic_rank import DynamicRankBlockTensor, DynamicRankFourGroupBlockLayout
from .model_categorical import CategoricalEntropyModel, CategoricalEntropyOutput
from .model_categorical_context_corruption import (
    CategoricalEntropyModelContextCorruption,
    ContextCorruptionEntropyOutput,
    build_codebook_knn,
)


class CategoricalEntropyModelDynamicRankReliable(
    CategoricalEntropyModelContextCorruption
):
    """Dynamic parity rank with generated cross-frame context blocked."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.layout = DynamicRankFourGroupBlockLayout(config.window_size)

    def _position_embeddings_for_blocks(
        self, blocks: DynamicRankBlockTensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time, group, local_y, local_x = self.layout.block_coordinates(blocks)
        target = (
            self.target_time(time)
            + self.target_group(group)
            + self.target_y(local_y)
            + self.target_x(local_x)
        )
        source = torch.zeros_like(target)
        source[:, 1:] = (
            self.source_time(time[:, :-1])
            + self.source_group(group[:, :-1])
            + self.source_y(local_y[:, :-1])
            + self.source_x(local_x[:, :-1])
        )
        return target.to(dtype), source.to(dtype)

    def _shifted_dynamic_inputs(
        self,
        blocks: DynamicRankBlockTensor,
        source_status: torch.Tensor | None = None,
    ) -> torch.Tensor:
        projected = self.code_projection(blocks.values)
        inputs = torch.empty_like(projected)
        inputs[:, 0] = self.bos.expand(projected.shape[0], -1, -1)
        inputs[:, 1:] = projected[:, :-1]
        if source_status is None:
            source_status = torch.zeros(
                blocks.values.shape[:2],
                dtype=torch.long,
                device=blocks.values.device,
            )
        if source_status.dtype != torch.int64 or tuple(source_status.shape) != tuple(
            blocks.values.shape[:2]
        ):
            raise ValueError(
                f"Expected int64 source status {tuple(blocks.values.shape[:2])}, "
                f"got {source_status.dtype} {tuple(source_status.shape)}"
            )
        inputs[:, 1:] += self.source_status_embedding(source_status[:, :-1])[
            :, :, None, :
        ]
        target, source = self._position_embeddings_for_blocks(blocks, projected.dtype)
        return inputs + target + source

    def _reliability_attention_mask(
        self,
        blocks: DynamicRankBlockTensor,
        source_status: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        num_blocks = blocks.values.shape[1]
        query_block = torch.arange(num_blocks, device=blocks.values.device)[:, None]
        key_block = torch.arange(num_blocks, device=blocks.values.device)[None, :]
        causal = key_block <= query_block
        target_time, _, _, _ = self.layout.block_coordinates(blocks)
        target_time = target_time[:, :, 0]
        key_source_time = torch.zeros_like(target_time)
        key_source_time[:, 1:] = target_time[:, :-1]
        key_generated = torch.zeros_like(source_status, dtype=torch.bool)
        key_generated[:, 1:] = source_status[:, :-1].bool()
        cross_frame_generated = key_generated[:, None, :] & (
            key_source_time[:, None, :] < target_time[:, :, None]
        )
        allowed = causal[None] & ~cross_frame_generated
        sites = self.layout.sites_per_block
        allowed = allowed.repeat_interleave(sites, dim=1).repeat_interleave(sites, dim=2)
        mask = torch.zeros(allowed.shape, dtype=dtype, device=blocks.values.device)
        mask.masked_fill_(~allowed, float("-inf"))
        return mask[:, None]

    def _transform_reliable(
        self,
        inputs: torch.Tensor,
        blocks: DynamicRankBlockTensor,
        source_status: torch.Tensor,
    ) -> torch.Tensor:
        hidden = inputs.reshape(
            inputs.shape[0],
            inputs.shape[1] * self.layout.sites_per_block,
            inputs.shape[-1],
        )
        mask = self._reliability_attention_mask(blocks, source_status, hidden.dtype)
        for layer in self.transformer.layers:
            hidden = layer(hidden, mask)
        return hidden.reshape_as(inputs)

    def predict(self, quantized: torch.Tensor):
        grid = quantized.permute(0, 2, 3, 4, 1).to(
            self.code_projection.weight.dtype
        )
        blocks = self.layout.to_blocks(grid)
        hidden = self.transformer(
            self._shifted_dynamic_inputs(blocks), self.layout.sites_per_block
        )
        return self.output_head(self.output_norm(hidden)), blocks

    def _targets(self, indices: torch.Tensor) -> torch.Tensor:
        return self.layout.to_blocks(indices.unsqueeze(-1)).values.squeeze(-1)

    def forward(
        self,
        quantized: torch.Tensor,
        indices: torch.Tensor,
        *,
        codebook: torch.Tensor | None = None,
        knn_indices: torch.Tensor | None = None,
        corrupted_samples: torch.Tensor | None = None,
        retained_groups: torch.Tensor | None = None,
        corrupt_loss_weight: float | None = None,
    ) -> CategoricalEntropyOutput | ContextCorruptionEntropyOutput:
        if codebook is None:
            return CategoricalEntropyModel.forward(self, quantized, indices)
        self._validate(quantized, indices)
        batch_size = indices.shape[0]
        if knn_indices is None:
            raise ValueError("knn_indices is required for context-corruption training")
        if corrupted_samples is None or tuple(corrupted_samples.shape) != (batch_size,):
            raise ValueError(f"Expected corrupted_samples [{batch_size}]")
        if retained_groups is None or tuple(retained_groups.shape) != (batch_size,):
            raise ValueError(f"Expected retained_groups [{batch_size}]")
        if corrupt_loss_weight is None or corrupt_loss_weight <= 0:
            raise ValueError("corrupt_loss_weight must be positive")

        code_grid = quantized.permute(0, 2, 3, 4, 1).to(
            self.code_projection.weight.dtype
        )
        code_blocks = self.layout.to_blocks(code_grid)
        index_blocks = self.layout.to_blocks(indices.unsqueeze(-1))
        targets = index_blocks.values.squeeze(-1)
        num_windows = code_blocks.num_windows
        sample_ids = torch.arange(batch_size, device=indices.device).repeat_interleave(
            num_windows
        )
        block_ranks = torch.arange(
            code_blocks.values.shape[1], device=indices.device
        ) % self.layout.num_groups
        corrupt_blocks = corrupted_samples[sample_ids, None] & (
            block_ranks[None, :] >= retained_groups[sample_ids, None]
        )
        corrupted_codes = code_blocks.values.clone()
        corrupted_targets = targets[corrupt_blocks]
        neighbor_choices = torch.randint(
            knn_indices.shape[1], corrupted_targets.shape, device=indices.device
        )
        replacement_indices = knn_indices[corrupted_targets, neighbor_choices]
        corrupted_codes[corrupt_blocks] = F.embedding(
            replacement_indices, codebook
        ).to(corrupted_codes.dtype)
        corrupted_blocks = DynamicRankBlockTensor(
            corrupted_codes,
            code_blocks.batch_size,
            code_blocks.latent_frames,
            code_blocks.height,
            code_blocks.width,
            code_blocks.windows_y,
            code_blocks.windows_x,
            code_blocks.rank_to_group,
        )
        hidden = self._transform_reliable(
            self._shifted_dynamic_inputs(corrupted_blocks, corrupt_blocks.long()),
            corrupted_blocks,
            corrupt_blocks.long(),
        )
        logits = self.output_head(self.output_norm(hidden))
        bits = F.cross_entropy(
            logits.float().reshape(-1, self.config.codebook_size),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape) / math.log(2.0)
        estimated_bits = bits.sum(dim=(-1, -2)).reshape(
            batch_size, num_windows
        ).sum(dim=1)
        indices_per_sample = indices[0].numel()
        clean_bits_per_index = estimated_bits[~corrupted_samples].sum() / (
            (~corrupted_samples).sum() * indices_per_sample
        )
        corrupt_bits_per_index = estimated_bits[corrupted_samples].sum() / (
            corrupted_samples.sum() * indices_per_sample
        )
        loss = clean_bits_per_index + corrupt_loss_weight * corrupt_bits_per_index
        return ContextCorruptionEntropyOutput(
            logits=logits,
            estimated_bits=estimated_bits,
            clean_bits_per_index=clean_bits_per_index,
            corrupt_bits_per_index=corrupt_bits_per_index,
            loss=loss,
        )
