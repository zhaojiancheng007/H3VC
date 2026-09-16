from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from entropy_model.model_categorical import (
    CategoricalEntropyModel,
    SoftCategoricalEntropyOutput,
)
from .stage1_vq import Stage1VQTokenizer


@dataclass(frozen=True)
class RelaxedQuantization:
    projected: torch.Tensor
    selected: torch.Tensor
    quantized: torch.Tensor
    indices: torch.Tensor
    target_probabilities: torch.Tensor


@dataclass(frozen=True)
class JointTokenizerOutput:
    reconstruction: torch.Tensor
    projected: torch.Tensor
    selected: torch.Tensor
    quantized: torch.Tensor
    indices: torch.Tensor


@dataclass(frozen=True)
class JointVQOutput:
    tokenizer: JointTokenizerOutput
    entropy: SoftCategoricalEntropyOutput


def selected_code_vectors(
    codebook: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    selected = F.embedding(indices, codebook)
    return selected.permute(0, 4, 1, 2, 3).contiguous()


def relaxed_quantize(
    projected: torch.Tensor,
    codebook: torch.Tensor,
    *,
    temperature: float,
) -> RelaxedQuantization:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if projected.ndim != 5 or projected.shape[1] != codebook.shape[1]:
        raise ValueError(
            "Projected/codebook shapes do not match: "
            f"{tuple(projected.shape)} and {tuple(codebook.shape)}"
        )
    with torch.autocast(device_type=projected.device.type, enabled=False):
        shape = projected.shape
        flat = projected.permute(0, 2, 3, 4, 1).reshape(-1, shape[1]).float()
        embeddings = codebook.float()
        squared_distances = torch.cdist(
            flat, embeddings, p=2, compute_mode="donot_use_mm_for_euclid_dist"
        ).square()
        flat_indices = squared_distances.argmin(dim=1)
        indices = flat_indices.reshape(shape[0], shape[2], shape[3], shape[4])
        selected = selected_code_vectors(embeddings, indices)
        projected_fp32 = projected.float()
        quantized = projected_fp32 + (selected - projected_fp32).detach()
        probabilities = torch.softmax(
            -squared_distances / temperature, dim=-1
        ).reshape(*indices.shape, embeddings.shape[0])
    return RelaxedQuantization(
        projected=projected_fp32,
        selected=selected,
        quantized=quantized,
        indices=indices,
        target_probabilities=probabilities,
    )


class JointVQModel(nn.Module):
    def __init__(
        self,
        tokenizer: Stage1VQTokenizer,
        entropy_model: CategoricalEntropyModel,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.entropy_model = entropy_model

    def forward(
        self, video: torch.Tensor, *, temperature: float, k: int
    ) -> JointVQOutput:
        latent = self.tokenizer.encode(video)
        projected = self.tokenizer.pre_quant(latent)
        relaxed = relaxed_quantize(
            projected,
            self.tokenizer.quantizer.embed,
            temperature=temperature,
        )
        reconstruction = self.tokenizer.decode(
            self.tokenizer.post_quant(relaxed.quantized.to(projected.dtype))
        )
        entropy = self.entropy_model.forward_soft(
            relaxed.quantized,
            relaxed.indices,
            relaxed.target_probabilities,
            k=k,
        )
        return JointVQOutput(
            tokenizer=JointTokenizerOutput(
                reconstruction=reconstruction,
                projected=relaxed.projected,
                selected=relaxed.selected,
                quantized=relaxed.quantized,
                indices=relaxed.indices,
            ),
            entropy=entropy,
        )
