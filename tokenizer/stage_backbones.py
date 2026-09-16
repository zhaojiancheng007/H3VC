from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class TokenizerSpec:
    name: str
    input_frames: int
    latent_channels: int
    codebook_dim: int


TOKENIZER_SPECS = {"minimax_h3": TokenizerSpec("minimax_h3", 107, 24, 24)}


def checkpoint_input_frames_match(tokenizer: str, checkpoint_frames: int) -> bool:
    if tokenizer == "minimax_h3":
        return checkpoint_frames in {90, 96, 107}
    return checkpoint_frames == TOKENIZER_SPECS[tokenizer].input_frames


def _training_checkpoint(owner, function, *args):
    if owner.training and getattr(owner, "gradient_checkpointing", False):
        return checkpoint(function, *args, use_reentrant=False)
    return function(*args)


class MiniMaxH3Backbone(nn.Module):
    MODEL_FRAMES = 107

    def __init__(self, model_dir: str | Path, *, training_frames: int = 107) -> None:
        super().__init__()
        self.training_frames = training_frames
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
        input_frames = int(video.shape[2])
        if self.training and input_frames != self.training_frames:
            raise ValueError(
                f"MiniMax-H3 training requires {self.training_frames} source frames, "
                f"got {input_frames}"
            )
        if input_frames not in {90, 96, self.MODEL_FRAMES}:
            raise ValueError(f"MiniMax-H3 cannot align {input_frames} input frames")
        if input_frames < self.MODEL_FRAMES:
            tail = video[:, :, -1:].expand(
                -1, -1, self.MODEL_FRAMES - input_frames, -1, -1
            )
            video = torch.cat((video, tail), dim=2)
        pixels = video.float().add(1.0).div(2.0)
        normalized = self.model.processor.transform_tensor(pixels)
        moments = self.model.encode_temporal(normalized)
        posterior_mean, _ = moments.chunk(2, dim=1)
        return posterior_mean

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        output_frames = self.training_frames if self.training else self.MODEL_FRAMES
        decoded = self.model.decode_base(
            latent.float(), frame_num=self.MODEL_FRAMES, process_image=False
        )
        if decoded.shape[2] != self.MODEL_FRAMES:
            raise RuntimeError(
                f"MiniMax-H3 decoded {decoded.shape[2]} frames, expected {self.MODEL_FRAMES}"
            )
        pixels = self.model.processor.revert_tensor(decoded)
        return pixels[:, :, :output_frames].mul(2.0).sub(1.0)


def build_backbone(
    tokenizer: str,
    *,

    minimax_h3_model_dir: str | Path,
    minimax_h3_training_frames: int | None = None,
) -> nn.Module:
    if tokenizer == "minimax_h3":
        training_frames = (
            TOKENIZER_SPECS[tokenizer].input_frames
            if minimax_h3_training_frames is None else minimax_h3_training_frames
        )
        return MiniMaxH3Backbone(minimax_h3_model_dir, training_frames=training_frames)
    raise ValueError(f"Unknown tokenizer {tokenizer!r}")
