from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WanVQStandardOutput:
    reconstruction: torch.Tensor
    indices: torch.Tensor
    encoder_latent: torch.Tensor
    projected: torch.Tensor
    selected_codebook: torch.Tensor
    quantized: torch.Tensor
    decoder_latent: torch.Tensor


class WanVQStandardAdapter(nn.Module):
    def __init__(
        self,
        wan_repo: str | Path,
        vae_checkpoint: str | Path,
        *,
        codebook_size: int = 4096,
        codebook_dim: int = 32,
    ) -> None:
        super().__init__()
        if codebook_size != 4096 or codebook_dim != 32:
            raise ValueError("Stage 1 requires one 4096x32 codebook")
        wan_repo = Path(wan_repo).resolve()
        vae_checkpoint = Path(vae_checkpoint).resolve()
        if not (wan_repo / "wan" / "modules" / "vae.py").is_file():
            raise FileNotFoundError(f"Official WanVAE source not found: {wan_repo}")
        if not vae_checkpoint.is_file():
            raise FileNotFoundError(vae_checkpoint)
        if str(wan_repo) not in sys.path:
            sys.path.insert(0, str(wan_repo))
        from wan.modules.vae import WanVAE

        wrapper = WanVAE(
            z_dim=16,
            vae_pth=str(vae_checkpoint),
            dtype=torch.float32,
            device="cpu",
        )
        self.wan = wrapper.model.train().requires_grad_(True)
        self.register_buffer("latent_mean", wrapper.mean.detach().clone())
        self.register_buffer("latent_inv_std", wrapper.std.detach().clone().reciprocal())

        self.pre_quant = nn.Conv3d(16, codebook_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.post_quant = nn.Conv3d(codebook_dim, 16, kernel_size=1)
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self._initialize_identity_projections()
        nn.init.normal_(self.codebook.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.codebook.weight.copy_(F.normalize(self.codebook.weight, p=2, dim=1))

    def _initialize_identity_projections(self) -> None:
        with torch.no_grad():
            self.pre_quant.weight.zero_()
            self.pre_quant.bias.zero_()
            self.post_quant.weight.zero_()
            self.post_quant.bias.zero_()
            identity = torch.eye(16)
            self.pre_quant.weight[:16, :, 0, 0, 0].copy_(identity)
            self.post_quant.weight[:, :16, 0, 0, 0].copy_(identity)

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        return self.wan.encode(
            video.float(), [self.latent_mean, self.latent_inv_std]
        )

    def project(self, encoder_latent: torch.Tensor) -> torch.Tensor:
        return self.pre_quant(encoder_latent)

    def quantize(
        self, projected: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if projected.ndim != 5 or projected.shape[1] != self.codebook_dim:
            raise ValueError(
                f"Expected projected [B,{self.codebook_dim},T,H,W], "
                f"got {tuple(projected.shape)}"
            )
        normalized = F.normalize(projected, p=2, dim=1)
        flat = normalized.permute(0, 2, 3, 4, 1).reshape(-1, self.codebook_dim)
        embeddings = F.normalize(self.codebook.weight, p=2, dim=1)
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            + embeddings.square().sum(dim=1).unsqueeze(0)
            - 2.0 * flat @ embeddings.t()
        )
        flat_indices = distances.argmin(dim=1)
        selected_flat = F.embedding(flat_indices, embeddings)
        selected = selected_flat.reshape(
            normalized.shape[0],
            normalized.shape[2],
            normalized.shape[3],
            normalized.shape[4],
            self.codebook_dim,
        ).permute(0, 4, 1, 2, 3).contiguous()
        quantized = normalized + (selected - normalized).detach()
        indices = flat_indices.reshape(
            normalized.shape[0],
            normalized.shape[2],
            normalized.shape[3],
            normalized.shape[4],
        )
        return quantized, normalized, selected, indices

    def forward(self, video: torch.Tensor) -> WanVQStandardOutput:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"Expected RGB video [B,3,T,H,W], got {tuple(video.shape)}")
        encoder_latent = self.encode(video)
        projected = self.project(encoder_latent)
        quantized, projected, selected, indices = self.quantize(projected)
        decoder_latent = self.post_quant(quantized)
        reconstruction = self.wan.decode(
            decoder_latent.float(), [self.latent_mean, self.latent_inv_std]
        )
        if reconstruction.shape != video.shape:
            raise RuntimeError(
                f"WanVQ reconstruction shape {tuple(reconstruction.shape)} "
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

    def training_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu() for name, value in self.state_dict().items()}

    def load_training_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected = set(self.state_dict())
        if set(state) != expected:
            raise ValueError(
                f"Standard model state fields are {sorted(state)}, expected {sorted(expected)}"
            )
        self.load_state_dict(state, strict=True)
