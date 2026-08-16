from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .stage1_vq import CODEBOOK_DIM, CODEBOOK_SIZE, Stage1VQTokenizer
from .stage1_vq_cosmos_cv888 import CosmosCV888Backbone


@dataclass(frozen=True)
class LoadedStage1Tokenizer:
    model: Stage1VQTokenizer
    name: str
    step: int
    checkpoint: Path


def load_stage1_tokenizer(
    checkpoint: str | Path,
    *,
    cosmos_repo: str | Path,
    cosmos_encoder: str | Path,
    cosmos_decoder: str | Path,
) -> LoadedStage1Tokenizer:
    checkpoint = Path(checkpoint).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {
        "version",
        "stage",
        "step",
        "tokenizer",
        "codebook_size",
        "codebook_dim",
        "model",
        "optimizer",
        "config",
    }
    if set(payload) != required:
        raise ValueError(
            f"Stage1 checkpoint fields are {sorted(payload)}, "
            f"expected {sorted(required)}"
        )
    if payload["version"] != 1 or payload["stage"] != 1:
        raise ValueError("Expected a version-1 Stage1 checkpoint")
    if payload["tokenizer"] != "cosmos_cv888":
        raise ValueError(f"Expected cosmos_cv888, got {payload['tokenizer']!r}")
    if (
        payload["codebook_size"] != CODEBOOK_SIZE
        or payload["codebook_dim"] != CODEBOOK_DIM
    ):
        raise ValueError(
            f"Expected codebook [{CODEBOOK_SIZE},{CODEBOOK_DIM}], got "
            f"[{payload['codebook_size']},{payload['codebook_dim']}]"
        )

    backbone = CosmosCV888Backbone(
        cosmos_repo, cosmos_encoder, cosmos_decoder
    )
    model = Stage1VQTokenizer(backbone)
    model.load_state_dict(payload["model"], strict=True)
    if not bool(model.quantizer.initted.item()):
        raise ValueError("Stage1 checkpoint contains an uninitialized codebook")
    return LoadedStage1Tokenizer(
        model=model,
        name=payload["tokenizer"],
        step=int(payload["step"]),
        checkpoint=checkpoint,
    )
