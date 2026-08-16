from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .model_quant_feature import CosmosFSQQuantFeatureEntropyModel


@dataclass(frozen=True)
class Stage3EntropyModelOutput:
    logits: tuple[torch.Tensor, ...]
    soft_estimated_bits: torch.Tensor
    soft_bits_per_site: torch.Tensor
    soft_bits_per_digit: torch.Tensor
    hard_estimated_bits: torch.Tensor
    hard_bits_per_site: torch.Tensor
    hard_bits_per_digit: torch.Tensor
    latent_shape: tuple[int, int, int]


class CosmosFSQStage3EntropyModel(CosmosFSQQuantFeatureEntropyModel):
    """Stage-3 entropy forward with differentiable per-digit FSQ targets."""

    def forward_soft(
        self,
        quant_features: torch.Tensor,
        target_indices: torch.Tensor,
        target_probabilities: tuple[torch.Tensor, ...],
    ) -> Stage3EntropyModelOutput:
        self._validate_feature_target_pair(quant_features, target_indices)
        if len(target_probabilities) != len(self.config.levels):
            raise ValueError(
                f"Expected {len(self.config.levels)} target distributions, "
                f"got {len(target_probabilities)}"
            )

        prediction = self.predict_quant_features(quant_features)
        soft_bit_tensors: list[torch.Tensor] = []
        hard_bit_tensors: list[torch.Tensor] = []
        hard_digits = self.fsq.indices_to_digits(target_indices)
        hard_digit_blocks = self.layout.to_blocks(hard_digits).values

        for dimension, (dimension_logits, probabilities, classes) in enumerate(
            zip(
                prediction.logits,
                target_probabilities,
                self.config.levels,
                strict=True,
            )
        ):
            expected_probability_shape = (*target_indices.shape, classes)
            if tuple(probabilities.shape) != expected_probability_shape:
                raise ValueError(
                    f"target_probabilities[{dimension}] has shape "
                    f"{tuple(probabilities.shape)}, expected {expected_probability_shape}"
                )
            probability_blocks = self.layout.to_blocks(probabilities).values
            log2_probabilities = (
                torch.log_softmax(dimension_logits.float(), dim=-1) / math.log(2.0)
            )
            soft_bit_tensors.append(
                -(probability_blocks * log2_probabilities).sum(dim=-1)
            )
            with torch.no_grad():
                hard_targets = hard_digit_blocks[..., dimension].unsqueeze(-1)
                hard_bit_tensors.append(
                    -log2_probabilities.detach().gather(-1, hard_targets).squeeze(-1)
                )

        batch, frames, height, width = map(int, target_indices.shape)
        site_count = batch * frames * height * width
        soft_bits_per_digit = torch.stack(
            [bits.sum() for bits in soft_bit_tensors]
        ) / site_count
        hard_bits_per_digit = torch.stack(
            [bits.sum() for bits in hard_bit_tensors]
        ) / site_count
        soft_per_window = torch.stack(soft_bit_tensors, dim=-1).sum(
            dim=(-1, -2, -3)
        )
        hard_per_window = torch.stack(hard_bit_tensors, dim=-1).sum(
            dim=(-1, -2, -3)
        )
        soft_estimated_bits = soft_per_window.reshape(
            batch, prediction.code_blocks.num_windows
        ).sum(dim=1)
        hard_estimated_bits = hard_per_window.reshape(
            batch, prediction.code_blocks.num_windows
        ).sum(dim=1)
        return Stage3EntropyModelOutput(
            logits=prediction.logits,
            soft_estimated_bits=soft_estimated_bits,
            soft_bits_per_site=soft_estimated_bits.sum() / site_count,
            soft_bits_per_digit=soft_bits_per_digit,
            hard_estimated_bits=hard_estimated_bits,
            hard_bits_per_site=hard_estimated_bits.sum() / site_count,
            hard_bits_per_digit=hard_bits_per_digit,
            latent_shape=(frames, height, width),
        )
