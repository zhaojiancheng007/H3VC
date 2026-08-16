from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model_standard import WanVQStandardAdapter, WanVQStandardOutput


@dataclass(frozen=True)
class WanVQStage3TokenizerOutput:
    tokenizer: WanVQStandardOutput
    target_probabilities: torch.Tensor


@dataclass(frozen=True)
class WanVQStage3Quantization:
    quantized: torch.Tensor
    projected: torch.Tensor
    selected_codebook: torch.Tensor
    indices: torch.Tensor
    target_probabilities: torch.Tensor


def quantize_with_relaxed_probabilities(
    projected: torch.Tensor,
    codebook: torch.Tensor,
    *,
    temperature: float,
) -> WanVQStage3Quantization:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if projected.ndim != 5 or codebook.ndim != 2:
        raise ValueError(
            f"Expected projected [B,C,T,H,W] and codebook [K,C], got "
            f"{tuple(projected.shape)} and {tuple(codebook.shape)}"
        )
    if projected.shape[1] != codebook.shape[1]:
        raise ValueError(
            f"Projected tensor {tuple(projected.shape)} does not match "
            f"codebook {tuple(codebook.shape)}"
        )

    # Keep the 4096-way distance and relaxed target distribution in FP32.
    # AMP is enabled around the surrounding Wan/entropy forward pass.
    with torch.autocast(device_type=projected.device.type, enabled=False):
        normalized = F.normalize(projected.float(), p=2, dim=1)
        embeddings = F.normalize(codebook.float(), p=2, dim=1)
        flat = normalized.permute(0, 2, 3, 4, 1).reshape(-1, normalized.shape[1])
        squared_distances = (
            flat.square().sum(dim=1, keepdim=True)
            + embeddings.square().sum(dim=1).unsqueeze(0)
            - 2.0 * flat @ embeddings.t()
        )

        flat_indices = squared_distances.argmin(dim=1)
        selected_flat = F.embedding(flat_indices, embeddings)
        selected = selected_flat.reshape(
            normalized.shape[0],
            normalized.shape[2],
            normalized.shape[3],
            normalized.shape[4],
            normalized.shape[1],
        ).permute(0, 4, 1, 2, 3).contiguous()
        quantized = normalized + (selected - normalized).detach()
        target_probabilities = torch.softmax(
            -squared_distances / temperature, dim=-1
        ).reshape(
            normalized.shape[0],
            normalized.shape[2],
            normalized.shape[3],
            normalized.shape[4],
            embeddings.shape[0],
        )
        indices = flat_indices.reshape(
            normalized.shape[0],
            normalized.shape[2],
            normalized.shape[3],
            normalized.shape[4],
        )
    return WanVQStage3Quantization(
        quantized=quantized,
        projected=normalized,
        selected_codebook=selected,
        indices=indices,
        target_probabilities=target_probabilities,
    )


class WanVQStage3Adapter(WanVQStandardAdapter):
    def forward(
        self, video: torch.Tensor, *, temperature: float
    ) -> WanVQStage3TokenizerOutput:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"Expected RGB video [B,3,T,H,W], got {tuple(video.shape)}")

        encoder_latent = self.encode(video)
        projected = self.project(encoder_latent)
        quantization = quantize_with_relaxed_probabilities(
            projected,
            self.codebook.weight,
            temperature=temperature,
        )
        decoder_latent = self.post_quant(quantization.quantized)
        reconstruction = self.wan.decode(
            decoder_latent.float(), [self.latent_mean, self.latent_inv_std]
        )
        if reconstruction.shape != video.shape:
            raise RuntimeError(
                f"WanVQ reconstruction shape {tuple(reconstruction.shape)} "
                f"does not match input {tuple(video.shape)}"
            )

        tokenizer_output = WanVQStandardOutput(
            reconstruction=reconstruction,
            indices=quantization.indices,
            encoder_latent=encoder_latent,
            projected=quantization.projected,
            selected_codebook=quantization.selected_codebook,
            quantized=quantization.quantized,
            decoder_latent=decoder_latent,
        )
        return WanVQStage3TokenizerOutput(
            tokenizer=tokenizer_output,
            target_probabilities=quantization.target_probabilities,
        )
