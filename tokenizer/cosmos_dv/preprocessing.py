from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PadInfo:
    frames: int
    height: int
    width: int
    padded_frames: int
    padded_height: int
    padded_width: int
    temporal_start: int
    height_start: int
    width_start: int


def _causal_aligned_length(length: int, alignment: int) -> int:
    if length <= 0 or alignment <= 0:
        raise ValueError(f"Invalid length/alignment: {length}/{alignment}")
    return 1 + ((length - 1 + alignment - 1) // alignment) * alignment


def pad_for_cosmos(
    video: torch.Tensor,
    *,
    spatial_alignment: int = 16,
    temporal_alignment: int = 8,
) -> tuple[torch.Tensor, PadInfo]:
    """Match Cosmos pad_video_batch for normalized [B,C,T,H,W] input."""
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError(f"Expected [B,3,T,H,W], got {tuple(video.shape)}")
    frames, height, width = map(int, video.shape[-3:])
    padded_frames = _causal_aligned_length(frames, temporal_alignment)
    padded_height = ((height + spatial_alignment - 1) // spatial_alignment) * spatial_alignment
    padded_width = ((width + spatial_alignment - 1) // spatial_alignment) * spatial_alignment

    pad_h = padded_height - height
    pad_w = padded_width - width
    if pad_h or pad_w:
        top, left = pad_h // 2, pad_w // 2
        # Official code pads uint8 with zero before mapping [0,255] to [-1,1].
        video = F.pad(
            video,
            (left, pad_w - left, top, pad_h - top, 0, 0),
            mode="constant",
            value=-1.0,
        )
    pad_t = padded_frames - frames
    temporal_start = pad_t // 2
    if pad_t:
        before = video[:, :, :1].expand(-1, -1, temporal_start, -1, -1)
        after = video[:, :, -1:].expand(-1, -1, pad_t - temporal_start, -1, -1)
        video = torch.cat([before, video, after], dim=2)
    return video, PadInfo(
        frames,
        height,
        width,
        padded_frames,
        padded_height,
        padded_width,
        temporal_start,
        pad_h // 2,
        pad_w // 2,
    )


def unpad_reconstruction(video: torch.Tensor, pad_info: PadInfo) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError(f"Expected [B,C,T,H,W], got {tuple(video.shape)}")
    expected = (pad_info.padded_frames, pad_info.padded_height, pad_info.padded_width)
    if tuple(video.shape[-3:]) != expected:
        raise ValueError(
            f"Decoder output shape {tuple(video.shape[-3:])} does not match padded input {expected}"
        )
    t0, h0, w0 = pad_info.temporal_start, pad_info.height_start, pad_info.width_start
    return video[
        :,
        :,
        t0 : t0 + pad_info.frames,
        h0 : h0 + pad_info.height,
        w0 : w0 + pad_info.width,
    ]
