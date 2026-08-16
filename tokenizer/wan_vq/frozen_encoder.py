from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .model_standard import WanVQStandardAdapter


class FrozenWanVQEncoder(nn.Module):
    """Inference-only index and code extractor for a trained WanVQ checkpoint."""

    codebook_size = 4096
    codebook_dim = 32

    def __init__(
        self,
        wan_repo: str | Path,
        vae_checkpoint: str | Path,
        tokenizer_checkpoint: str | Path,
        *,
        device: str | torch.device,
    ) -> None:
        super().__init__()
        payload = torch.load(tokenizer_checkpoint, map_location="cpu", weights_only=False)
        required = {"model", "step", "model_config"}
        missing = required - set(payload)
        if missing:
            raise ValueError(f"Tokenizer checkpoint is missing fields: {sorted(missing)}")
        if payload["step"] != 10000:
            raise ValueError(f"Expected WanVQ 10k checkpoint, got step={payload['step']}")
        if payload["model_config"] != {
            "codebook_size": self.codebook_size,
            "codebook_dim": self.codebook_dim,
            "wan_latent_channels": 16,
            "normalized_lookup": True,
        }:
            raise ValueError(f"Unexpected WanVQ model_config: {payload['model_config']}")
        model = WanVQStandardAdapter(
            wan_repo,
            vae_checkpoint,
            codebook_size=self.codebook_size,
            codebook_dim=self.codebook_dim,
        )
        model.load_training_state_dict(payload["model"])
        self.model = model.to(device).eval()
        self.model.requires_grad_(False)
        self.checkpoint = str(Path(tokenizer_checkpoint).resolve())
        self.step = int(payload["step"])

    @torch.inference_mode()
    def encode(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"Expected video [B,3,T,H,W], got {tuple(video.shape)}")
        encoder_latent = self.model.encode(video)
        projected = self.model.project(encoder_latent)
        _, _, selected, indices = self.model.quantize(projected)
        if indices.ndim != 4 or selected.ndim != 5:
            raise RuntimeError(
                f"Unexpected WanVQ outputs: indices={tuple(indices.shape)}, "
                f"selected_codebook={tuple(selected.shape)}"
            )
        if int(indices.min()) < 0 or int(indices.max()) >= self.codebook_size:
            raise RuntimeError("WanVQ produced an out-of-range index")
        return indices, selected
