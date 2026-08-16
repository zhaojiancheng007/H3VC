from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from entropy_model.model_wan_vq_categorical_stage3 import (
    WanVQCategoricalStage3EntropyModel,
    WanVQStage3EntropyOutput,
)
from tokenizer.wan_vq.model_stage3 import WanVQStage3Adapter
from tokenizer.wan_vq.model_standard import WanVQStandardOutput


@dataclass(frozen=True)
class WanVQJointRDOutput:
    tokenizer: WanVQStandardOutput
    entropy: WanVQStage3EntropyOutput


def estimated_video_bpp(estimated_bits: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
    pixels = video.shape[0] * video.shape[2] * video.shape[3] * video.shape[4]
    return estimated_bits.sum().float() / pixels


class WanVQJointRDModel(nn.Module):
    def __init__(
        self,
        tokenizer: WanVQStage3Adapter,
        entropy_model: WanVQCategoricalStage3EntropyModel,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.entropy_model = entropy_model

    def forward(self, video: torch.Tensor, *, temperature: float) -> WanVQJointRDOutput:
        tokenizer_result = self.tokenizer(video, temperature=temperature)
        tokenized = tokenizer_result.tokenizer
        entropy = self.entropy_model.forward_soft(
            tokenized.quantized,
            tokenized.indices,
            tokenizer_result.target_probabilities,
        )
        return WanVQJointRDOutput(tokenizer=tokenized, entropy=entropy)
