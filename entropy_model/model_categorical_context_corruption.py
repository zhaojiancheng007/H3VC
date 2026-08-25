from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .model_categorical import CategoricalEntropyModel, CategoricalEntropyOutput


@dataclass(frozen=True)
class ContextCorruptionEntropyOutput:
    logits: torch.Tensor
    estimated_bits: torch.Tensor
    clean_bits_per_index: torch.Tensor
    corrupt_bits_per_index: torch.Tensor
    loss: torch.Tensor


def build_codebook_knn(codebook: torch.Tensor, k: int) -> torch.Tensor:
    if codebook.ndim != 2:
        raise ValueError(f"Expected codebook [K,C], got {tuple(codebook.shape)}")
    if k <= 0 or k >= codebook.shape[0]:
        raise ValueError(f"k must be in [1,{codebook.shape[0] - 1}], got {k}")
    distances = torch.cdist(codebook.float(), codebook.float(), p=2)
    distances.fill_diagonal_(torch.inf)
    return distances.topk(k, dim=1, largest=False).indices


class CategoricalEntropyModelContextCorruption(CategoricalEntropyModel):
    """Categorical entropy model trained with simulated generated contexts."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.source_status_embedding = nn.Embedding(2, config.d_model)
        nn.init.zeros_(self.source_status_embedding.weight)

    def _shifted_inputs(
        self,
        code_blocks: torch.Tensor,
        source_status: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs = super()._shifted_inputs(code_blocks)
        if source_status is None:
            source_status = torch.zeros(
                code_blocks.shape[:2], dtype=torch.long, device=code_blocks.device
            )
        expected = code_blocks.shape[:2]
        if source_status.dtype != torch.int64 or tuple(source_status.shape) != expected:
            raise ValueError(
                f"Expected int64 source status {expected}, got "
                f"{source_status.dtype} {tuple(source_status.shape)}"
            )
        status_inputs = torch.zeros_like(inputs)
        status_inputs[:, 1:] = self.source_status_embedding(source_status[:, :-1])[
            :, :, None, :
        ]
        return inputs + status_inputs


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
            return super().forward(quantized, indices)
        self._validate(quantized, indices)
        batch_size = indices.shape[0]
        if knn_indices is None:
            raise ValueError("knn_indices is required for context-corruption training")
        if tuple(codebook.shape) != (
            self.config.codebook_size,
            self.config.codebook_dim,
        ):
            raise ValueError(
                f"Expected codebook [{self.config.codebook_size},"
                f"{self.config.codebook_dim}], got {tuple(codebook.shape)}"
            )
        if knn_indices.ndim != 2 or knn_indices.shape[0] != self.config.codebook_size:
            raise ValueError(
                f"Expected kNN indices [{self.config.codebook_size},k], "
                f"got {tuple(knn_indices.shape)}"
            )
        if (
            corrupted_samples is None
            or corrupted_samples.dtype != torch.bool
            or tuple(corrupted_samples.shape) != (batch_size,)
        ):
            raise ValueError(f"Expected bool corrupted_samples [{batch_size}]")
        if (
            retained_groups is None
            or retained_groups.dtype != torch.int64
            or tuple(retained_groups.shape) != (batch_size,)
        ):
            raise ValueError(f"Expected int64 retained_groups [{batch_size}]")
        if corrupt_loss_weight is None or corrupt_loss_weight <= 0.0:
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
        block_groups = torch.arange(
            code_blocks.values.shape[1], device=indices.device
        ) % self.layout.num_groups
        corrupt_blocks = corrupted_samples[sample_ids, None] & (
            block_groups[None, :] >= retained_groups[sample_ids, None]
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
        hidden = self.transformer(
            self._shifted_inputs(corrupted_codes, corrupt_blocks.long()),
            self.layout.sites_per_block,
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
