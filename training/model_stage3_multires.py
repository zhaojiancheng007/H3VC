from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from entropy_model.model_quant_feature_stage3 import (
    CosmosFSQStage3EntropyModel,
    Stage3EntropyModelOutput,
)
from tokenizer.cosmos_dv.preprocessing import pad_for_cosmos, unpad_reconstruction
from tokenizer.cosmos_dv.trainable_stage3_checkpointed import (
    CheckpointedTrainableCosmosDV,
    fsq_surrogate_probabilities,
)
from tokenizer.cosmos_dv.trainable_stage3_native_checkpointed import (
    NativeCheckpointedTrainableCosmosDV,
)


@dataclass(frozen=True)
class JointRDOutput:
    reconstruction: torch.Tensor
    indices: torch.Tensor
    quant_features: torch.Tensor
    entropy: Stage3EntropyModelOutput


def estimated_video_bpp(
    estimated_bits: torch.Tensor,
    video: torch.Tensor,
) -> torch.Tensor:
    pixel_count = video.shape[0] * video.shape[2] * video.shape[3] * video.shape[4]
    return estimated_bits.sum().float() / pixel_count


class CosmosDVJointRDModel(nn.Module):
    def __init__(
        self,
        tokenizer: (
            CheckpointedTrainableCosmosDV
            | NativeCheckpointedTrainableCosmosDV
        ),
        entropy_model: CosmosFSQStage3EntropyModel,
        *,
        surrogate_temperature: float,
    ) -> None:
        super().__init__()
        if surrogate_temperature <= 0:
            raise ValueError(
                f"surrogate_temperature must be positive, got {surrogate_temperature}"
            )
        self.tokenizer = tokenizer
        self.entropy_model = entropy_model
        self.surrogate_temperature = surrogate_temperature

    def forward(self, video: torch.Tensor) -> JointRDOutput:
        padded, pad_info = pad_for_cosmos(video, temporal_alignment=4)
        tokenized = self.tokenizer(padded)
        target_probabilities = fsq_surrogate_probabilities(
            tokenized.bounded,
            temperature=self.surrogate_temperature,
            levels=tuple(self.entropy_model.config.levels),
        )
        entropy = self.entropy_model.forward_soft(
            tokenized.quant_features,
            tokenized.indices,
            target_probabilities,
        )
        return JointRDOutput(
            reconstruction=unpad_reconstruction(
                tokenized.reconstruction, pad_info
            ),
            indices=tokenized.indices,
            quant_features=tokenized.quant_features,
            entropy=entropy,
        )
