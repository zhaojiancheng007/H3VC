from __future__ import annotations

import torch

from .model_standard import WanVQStandardAdapter, WanVQStandardOutput
from .parallel_wan import parallel_decode, parallel_encode


class ParallelBF16WanVQStandardAdapter(WanVQStandardAdapter):
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        return parallel_encode(
            self.wan,
            video.float(),
            [self.latent_mean, self.latent_inv_std],
        )

    def quantize(
        self, projected: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Autocast must not change the 4096-way nearest-neighbor assignment.
        with torch.autocast(device_type=projected.device.type, enabled=False):
            return super().quantize(projected.float())

    def forward(self, video: torch.Tensor) -> WanVQStandardOutput:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"Expected RGB video [B,3,T,H,W], got {tuple(video.shape)}")

        encoder_latent = self.encode(video)
        projected = self.project(encoder_latent)
        quantized, projected, selected, indices = self.quantize(projected)
        decoder_latent = self.post_quant(quantized)
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
        return WanVQStandardOutput(
            reconstruction=reconstruction,
            indices=indices,
            encoder_latent=encoder_latent,
            projected=projected,
            selected_codebook=selected,
            quantized=quantized,
            decoder_latent=decoder_latent,
        )
