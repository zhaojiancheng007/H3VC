from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .stage1_vq import CosmosCV488Backbone, WanVAEBackbone
from .stage1_vq_cosmos_cv888 import CosmosCV888Backbone


@dataclass(frozen=True)
class TokenizerSpec:
    name: str
    input_frames: int
    latent_channels: int
    codebook_dim: int


TOKENIZER_SPECS = {
    "wanvae": TokenizerSpec("wanvae", 97, 16, 16),
    "cosmos_cv488": TokenizerSpec("cosmos_cv488", 97, 16, 16),
    "cosmos_cv888": TokenizerSpec("cosmos_cv888", 97, 16, 16),
    "minimax_h3": TokenizerSpec("minimax_h3", 96, 24, 24),
}


def checkpoint_input_frames_match(tokenizer: str, checkpoint_frames: int) -> bool:
    current_frames = TOKENIZER_SPECS[tokenizer].input_frames
    return checkpoint_frames == current_frames or (
        tokenizer == "minimax_h3"
        and checkpoint_frames == 90
        and current_frames == 96
    )


def _training_checkpoint(owner, function, *args):
    if owner.training and getattr(owner, "gradient_checkpointing", False):
        return checkpoint(function, *args, use_reentrant=False)
    return function(*args)


class MiniMaxH3Backbone(nn.Module):
    MODEL_FRAMES = 107

    def __init__(self, model_dir: str | Path) -> None:
        super().__init__()
        model_dir = Path(model_dir).resolve()
        if not (model_dir / "minimax_h3_video_vae.py").is_file():
            raise FileNotFoundError(model_dir / "minimax_h3_video_vae.py")
        sys.path.insert(0, str(model_dir.parent))
        from video_vae.minimax_h3_video_vae import MiniMaxH3VideoVAE
        from video_vae import vae_cnn, vae_vit

        vae_cnn.maybe_checkpoint = _training_checkpoint
        vae_vit.maybe_checkpoint = _training_checkpoint

        self.model = MiniMaxH3VideoVAE.from_pretrained(str(model_dir)).model
        if self.model.vae_ratio != 16 or self.model.vae_ratio_t != 4:
            raise RuntimeError(
                f"Unexpected H3 downsampling: temporal={self.model.vae_ratio_t}, "
                f"spatial={self.model.vae_ratio}"
            )
        self.model.apply(
            lambda module: self.model._set_gradient_checkpointing(module, True)
        )

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        source_frames = TOKENIZER_SPECS["minimax_h3"].input_frames
        if video.shape[2] != source_frames:
            raise ValueError(
                f"MiniMax-H3 requires {source_frames} source frames, got {video.shape[2]}"
            )
        tail = video[:, :, -1:].expand(
            -1, -1, self.MODEL_FRAMES - source_frames, -1, -1
        )
        model_input = torch.cat((video, tail), dim=2)
        pixels = model_input.float().add(1.0).div(2.0)
        normalized = self.model.processor.transform_tensor(pixels)
        moments = self.model.encode_temporal(normalized)
        posterior_mean, _ = moments.chunk(2, dim=1)
        return posterior_mean

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        source_frames = TOKENIZER_SPECS["minimax_h3"].input_frames
        decoded = self.model.decode_base(
            latent.float(), frame_num=self.MODEL_FRAMES, process_image=False
        )
        if decoded.shape[2] != self.MODEL_FRAMES:
            raise RuntimeError(
                f"MiniMax-H3 decoded {decoded.shape[2]} frames, expected {self.MODEL_FRAMES}"
            )
        pixels = self.model.processor.revert_tensor(decoded)
        return pixels[:, :, :source_frames].mul(2.0).sub(1.0)


def build_backbone(
    tokenizer: str,
    *,
    wan_repo: str | Path,
    wan_checkpoint: str | Path,
    cosmos_repo: str | Path,
    cosmos_cv488_encoder: str | Path,
    cosmos_cv488_decoder: str | Path,
    cosmos_cv888_encoder: str | Path,
    cosmos_cv888_decoder: str | Path,
    minimax_h3_model_dir: str | Path,
) -> nn.Module:
    if tokenizer == "wanvae":
        return WanVAEBackbone(wan_repo, wan_checkpoint)
    if tokenizer == "cosmos_cv488":
        return CosmosCV488Backbone(
            cosmos_repo, cosmos_cv488_encoder, cosmos_cv488_decoder
        )
    if tokenizer == "cosmos_cv888":
        return CosmosCV888Backbone(
            cosmos_repo, cosmos_cv888_encoder, cosmos_cv888_decoder
        )
    if tokenizer == "minimax_h3":
        return MiniMaxH3Backbone(minimax_h3_model_dir)
    raise ValueError(f"Unknown tokenizer {tokenizer!r}")
