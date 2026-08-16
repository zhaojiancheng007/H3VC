from __future__ import annotations

import torch

from .model_stage3 import (
    WanVQStage3TokenizerOutput,
    quantize_with_relaxed_probabilities,
)
from .model_standard import WanVQStandardAdapter, WanVQStandardOutput
from .parallel_wan import parallel_decode, parallel_encode


class ParallelWanVQStage3Adapter(WanVQStandardAdapter):
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        return parallel_encode(
            self.wan,
            video.float(),
            [self.latent_mean, self.latent_inv_std],
        )

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
        reconstruction = parallel_decode(
            self.wan,
            decoder_latent.float(),
            [self.latent_mean, self.latent_inv_std],
        )
        if reconstruction.shape != video.shape:
            raise RuntimeError(
                f"Parallel WanVQ reconstruction shape {tuple(reconstruction.shape)} "
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
