from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .layout import BlockTensor
from .model import CosmosFSQEntropyModel, EntropyModelOutput


@dataclass(frozen=True)
class QuantFeaturePrediction:
    logits: tuple[torch.Tensor, ...]
    code_blocks: BlockTensor


class CosmosFSQQuantFeatureEntropyModel(CosmosFSQEntropyModel):
    """Entropy model whose explicit input is the decoded six-dimensional FSQ code."""

    def _validate_quant_features(self, quant_features: torch.Tensor) -> None:
        if quant_features.ndim != 5 or quant_features.shape[1] != self.fsq.dimensions:
            raise ValueError(
                "Expected quant_features [B,6,T,H,W], got "
                f"{tuple(quant_features.shape)}"
            )
        if not quant_features.is_floating_point():
            raise TypeError(
                f"quant_features must be floating point, got {quant_features.dtype}"
            )
        if not torch.isfinite(quant_features).all().item():
            raise FloatingPointError("quant_features contain non-finite values")

    def _validate_feature_target_pair(
        self, quant_features: torch.Tensor, target_indices: torch.Tensor
    ) -> None:
        self._validate_quant_features(quant_features)
        if target_indices.ndim != 4:
            raise ValueError(
                f"Expected target_indices [B,T,H,W], got {tuple(target_indices.shape)}"
            )
        expected_shape = (
            target_indices.shape[0],
            self.fsq.dimensions,
            *target_indices.shape[1:],
        )
        if tuple(quant_features.shape) != expected_shape:
            raise ValueError(
                f"quant_features shape {tuple(quant_features.shape)} does not match "
                f"target_indices shape {tuple(target_indices.shape)}"
            )
        expected_features = self.fsq.indices_to_codes(
            target_indices, dtype=quant_features.dtype
        ).permute(0, 4, 1, 2, 3)
        if not torch.equal(quant_features, expected_features):
            maximum_error = float(
                (quant_features.float() - expected_features.float()).abs().max().item()
            )
            raise ValueError(
                "quant_features do not represent target_indices; "
                f"maximum absolute error is {maximum_error}"
            )

    def predict_quant_features(
        self, quant_features: torch.Tensor
    ) -> QuantFeaturePrediction:
        self._validate_quant_features(quant_features)
        code_grid = quant_features.permute(0, 2, 3, 4, 1).to(
            dtype=self.code_projection.weight.dtype
        )
        code_blocks = self.layout.to_blocks(code_grid)
        return QuantFeaturePrediction(
            logits=self.forward_blocks(code_blocks.values),
            code_blocks=code_blocks,
        )

    def forward(
        self, quant_features: torch.Tensor, target_indices: torch.Tensor
    ) -> EntropyModelOutput:
        self._validate_feature_target_pair(quant_features, target_indices)
        prediction = self.predict_quant_features(quant_features)
        digits = self.fsq.indices_to_digits(target_indices)
        digit_blocks = self.layout.to_blocks(digits)
        if tuple(digit_blocks.values.shape[:3]) != tuple(
            prediction.code_blocks.values.shape[:3]
        ):
            raise RuntimeError("Quant-feature and target block layouts disagree")

        bit_tensors: list[torch.Tensor] = []
        for dimension, dimension_logits in enumerate(prediction.logits):
            targets = digit_blocks.values[..., dimension]
            bits = F.cross_entropy(
                dimension_logits.float().reshape(-1, self.config.levels[dimension]),
                targets.reshape(-1),
                reduction="none",
            ).reshape(targets.shape) / math.log(2.0)
            bit_tensors.append(bits)

        batch, frames, height, width = map(int, target_indices.shape)
        bits_per_digit = torch.stack([bits.sum() for bits in bit_tensors]) / (
            batch * frames * height * width
        )
        total_per_window = torch.stack(bit_tensors, dim=-1).sum(
            dim=(-1, -2, -3)
        )
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
