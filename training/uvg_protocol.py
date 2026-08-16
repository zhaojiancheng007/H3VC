from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tokenizer.cosmos_dv.preprocessing import PadInfo, pad_for_cosmos, unpad_reconstruction
from tokenizer.stage_backbones import TOKENIZER_SPECS


UVG_METRIC_FRAMES = 96


@dataclass(frozen=True)
class EvaluationInput:
    reference: torch.Tensor
    model_input: torch.Tensor
    model_frames: int
    cosmos_pad: PadInfo | None


def uvg_metric_frames(tokenizer: str) -> int:
    return min(UVG_METRIC_FRAMES, TOKENIZER_SPECS[tokenizer].input_frames)


def uvg_source_frames(tokenizer: str) -> int:
    if tokenizer == "minimax_h3":
        return uvg_metric_frames(tokenizer)
    return TOKENIZER_SPECS[tokenizer].input_frames


def prepare_uvg_input(video: torch.Tensor, tokenizer: str) -> EvaluationInput:
    spec = TOKENIZER_SPECS[tokenizer]
    expected_frames = uvg_source_frames(tokenizer)
    if video.ndim != 4 or video.shape[0] != expected_frames:
        raise ValueError(
            f"Expected [{expected_frames},3,H,W] {tokenizer} input, "
            f"got {tuple(video.shape)}"
        )
    reference = video[:uvg_metric_frames(tokenizer)]
    if tokenizer == "minimax_h3":
        height = (video.shape[-2] + 15) // 16 * 16
        width = (video.shape[-1] + 15) // 16 * 16
        video = F.pad(
            video,
            (0, width - video.shape[-1], 0, height - video.shape[-2]),
            mode="replicate",
        )
    model_input = video.permute(1, 0, 2, 3).unsqueeze(0).mul(2.0).sub(1.0)
    cosmos_pad = None
    if tokenizer in {"cosmos_cv488", "cosmos_cv888"}:
        alignment = 4 if tokenizer == "cosmos_cv488" else 8
        model_input, cosmos_pad = pad_for_cosmos(
            model_input, temporal_alignment=alignment, spatial_alignment=16
        )
    if model_input.shape[2] != spec.input_frames:
        raise RuntimeError(
            f"{tokenizer} must forward {spec.input_frames} frames, "
            f"got {model_input.shape[2]}"
        )
    return EvaluationInput(reference, model_input, spec.input_frames, cosmos_pad)


def finish_uvg_reconstruction(
    reconstruction: torch.Tensor, prepared: EvaluationInput
) -> torch.Tensor:
    if prepared.cosmos_pad is not None:
        reconstruction = unpad_reconstruction(reconstruction, prepared.cosmos_pad)
    if reconstruction.shape[2] != prepared.model_frames:
        raise RuntimeError(
            f"Decoder must reconstruct {prepared.model_frames} frames before "
            f"metric cropping, got {reconstruction.shape[2]}"
        )
    reconstruction = reconstruction[:, :, : prepared.reference.shape[0]]
    reconstruction = reconstruction[
        :, :, :, : prepared.reference.shape[-2], : prepared.reference.shape[-1]
    ]
    return (
        reconstruction.float()
        .clamp(-1.0, 1.0)
        .add(1.0)
        .div(2.0)
        .squeeze(0)
        .permute(1, 0, 2, 3)
        .cpu()
    )
