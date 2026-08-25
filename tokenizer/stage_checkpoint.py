from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .stage1_vq import CODEBOOK_SIZE, QUANTIZER_GEOMETRY, Stage1VQTokenizer
from .stage_backbones import (
    TOKENIZER_SPECS,
    build_backbone,
    checkpoint_input_frames_match,
)


@dataclass(frozen=True)
class LoadedStage1Tokenizer:
    model: Stage1VQTokenizer
    name: str
    step: int
    checkpoint: Path
    input_frames: int
    latent_channels: int
    codebook_dim: int


def load_stage1_tokenizer(
    checkpoint: str | Path,
    *,
    wan_repo: str | Path,
    wan_checkpoint: str | Path,
    cosmos_repo: str | Path,
    cosmos_encoder: str | Path,
    cosmos_decoder: str | Path,
    cosmos_cv888_encoder: str | Path,
    cosmos_cv888_decoder: str | Path,
    minimax_h3_model_dir: str | Path,
) -> LoadedStage1Tokenizer:
    checkpoint = Path(checkpoint).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    common = {
        "version",
        "stage",
        "step",
        "tokenizer",
        "input_frames",
        "latent_channels",
        "quantizer_geometry",
        "codebook_size",
        "codebook_dim",
        "model",
        "optimizer",
        "config",
    }
    version = int(payload.get("version", -1))
    required = common if version == 2 else common | {"training_state", "rng_states"}
    if set(payload) != required:
        raise ValueError(
            f"Stage1 checkpoint fields are {sorted(payload)}, "
            f"expected version-{version} fields {sorted(required)}"
        )
    if version not in (2, 3) or payload["stage"] != 1:
        raise ValueError("Stage2/3 training requires a version-2 or version-3 Stage1 checkpoint")
    name = payload["tokenizer"]
    spec = TOKENIZER_SPECS[name]
    if not checkpoint_input_frames_match(name, payload["input_frames"]):
        raise ValueError(
            f"Stage1 checkpoint input frames {payload['input_frames']} != "
            f"expected {spec.input_frames}"
        )
    geometry = (
        payload["latent_channels"],
        payload["codebook_size"],
        payload["codebook_dim"],
        payload["quantizer_geometry"],
    )
    expected = (
        spec.latent_channels,
        CODEBOOK_SIZE,
        spec.codebook_dim,
        QUANTIZER_GEOMETRY,
    )
    if geometry != expected:
        raise ValueError(f"Stage1 checkpoint geometry {geometry} != expected {expected}")

    backbone = build_backbone(
        name,
        wan_repo=wan_repo,
        wan_checkpoint=wan_checkpoint,
        cosmos_repo=cosmos_repo,
        cosmos_cv488_encoder=cosmos_encoder,
        minimax_h3_training_frames=int(payload["input_frames"]),
        cosmos_cv488_decoder=cosmos_decoder,
        cosmos_cv888_encoder=cosmos_cv888_encoder,
        cosmos_cv888_decoder=cosmos_cv888_decoder,
        minimax_h3_model_dir=minimax_h3_model_dir,
    )
    model = Stage1VQTokenizer(
        backbone,
        latent_channels=spec.latent_channels,
        codebook_dim=spec.codebook_dim,
    )
    model.load_state_dict(payload["model"], strict=True)
    if not bool(model.quantizer.initted.item()):
        raise ValueError("Stage1 checkpoint contains an uninitialized codebook")
    return LoadedStage1Tokenizer(
        model=model,
        name=name,
        step=int(payload["step"]),
        checkpoint=checkpoint,
        input_frames=int(payload["input_frames"]),
        latent_channels=spec.latent_channels,
        codebook_dim=spec.codebook_dim,
    )
