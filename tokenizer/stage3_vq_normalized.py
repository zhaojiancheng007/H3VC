from __future__ import annotations

import torch
import torch.nn.functional as F

from .stage3_vq import (
    JointTokenizerOutput,
    JointVQModel,
    JointVQOutput,
    RelaxedQuantization,
    selected_code_vectors,
)


def normalized_relaxed_quantize(
    projected: torch.Tensor,
    codebook: torch.Tensor,
    *,
    temperature: float,
) -> RelaxedQuantization:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if projected.ndim != 5 or projected.shape[1] != codebook.shape[1]:
        raise ValueError(
            f"Projected/codebook shapes do not match: "
            f"{tuple(projected.shape)} and {tuple(codebook.shape)}"
        )
    with torch.autocast(device_type=projected.device.type, enabled=False):
        normalized = F.normalize(projected.float(), p=2, dim=1)
        embeddings = F.normalize(codebook.float(), p=2, dim=1)
        shape = normalized.shape
        flat = normalized.permute(0, 2, 3, 4, 1).reshape(-1, shape[1])
        squared_distances = torch.cdist(
            flat, embeddings, p=2, compute_mode="donot_use_mm_for_euclid_dist"
        ).square()
        flat_indices = squared_distances.argmin(dim=1)
        indices = flat_indices.reshape(shape[0], shape[2], shape[3], shape[4])
        selected = selected_code_vectors(embeddings, indices)
        quantized = normalized + (selected - normalized).detach()
        probabilities = torch.softmax(-squared_distances / temperature, dim=-1).reshape(
            *indices.shape, embeddings.shape[0]
        )
    return RelaxedQuantization(
        projected=normalized,
        selected=selected,
        quantized=quantized,
        indices=indices,
        target_probabilities=probabilities,
    )


class NormalizedJointVQModel(JointVQModel):
    def forward(self, video: torch.Tensor, *, temperature: float) -> JointVQOutput:
        latent = self.tokenizer.encode(video)
        projected = self.tokenizer.pre_quant(latent)
        relaxed = normalized_relaxed_quantize(
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
