from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .model_wan_vq_categorical import WanVQCategoricalEntropyModel


@dataclass(frozen=True)
class WanVQStage3EntropyOutput:
    logits: torch.Tensor
    soft_estimated_bits: torch.Tensor
    hard_estimated_bits: torch.Tensor
    soft_bits_per_index: torch.Tensor
    hard_bits_per_index: torch.Tensor
    latent_shape: tuple[int, int, int]


class WanVQCategoricalStage3EntropyModel(WanVQCategoricalEntropyModel):
    def forward_soft(
        self,
        quant_features: torch.Tensor,
        target_indices: torch.Tensor,
        target_probabilities: torch.Tensor,
    ) -> WanVQStage3EntropyOutput:
        self._validate_pair(quant_features, target_indices)
        expected = (*target_indices.shape, self.config.codebook_size)
        if tuple(target_probabilities.shape) != expected:
            raise ValueError(
                f"Expected target probabilities {expected}, got {tuple(target_probabilities.shape)}"
            )

        logits, code_blocks = self.predict_quant_features(quant_features)
        targets = self.layout.to_blocks(target_indices.unsqueeze(-1)).values.squeeze(-1)
        # The transformer runs under BF16 autocast, while probabilities and
        # bit estimates retain the FP32 Stage3 objective.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            probability_blocks = self.layout.to_blocks(target_probabilities.float()).values
            log2_probabilities = torch.log_softmax(logits.float(), dim=-1) / math.log(2.0)
            soft_bits = -(probability_blocks * log2_probabilities).sum(dim=-1)
            hard_bits = -log2_probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        batch, frames, height, width = map(int, target_indices.shape)
        index_count = batch * frames * height * width
        soft_estimated_bits = soft_bits.sum(dim=(-1, -2)).reshape(
            batch, code_blocks.num_windows
        ).sum(dim=1)
        hard_estimated_bits = hard_bits.sum(dim=(-1, -2)).reshape(
            batch, code_blocks.num_windows
        ).sum(dim=1)
        return WanVQStage3EntropyOutput(
            logits=logits,
            soft_estimated_bits=soft_estimated_bits,
            hard_estimated_bits=hard_estimated_bits,
            soft_bits_per_index=soft_estimated_bits.sum() / index_count,
            hard_bits_per_index=hard_estimated_bits.sum() / index_count,
            latent_shape=(frames, height, width),
        )
