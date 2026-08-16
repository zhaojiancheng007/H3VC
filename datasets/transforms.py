from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def resize_short_side(video: torch.Tensor, short_side: int) -> torch.Tensor:
    """Resize uint8/float [T,C,H,W], preserving aspect ratio."""
    if video.ndim != 4:
        raise ValueError(f"Expected [T,C,H,W], got {tuple(video.shape)}")
    if short_side <= 0:
        raise ValueError(f"short_side must be positive, got {short_side}")
    height, width = video.shape[-2:]
    scale = short_side / min(height, width)
    new_height = int(math.floor(height * scale + 0.5))
    new_width = int(math.floor(width * scale + 0.5))
    resized = F.interpolate(
        video.float(), size=(new_height, new_width), mode="bicubic", align_corners=False
    )
    return resized.clamp(0, 255)


def shared_crop(
    video: torch.Tensor,
    crop_size: int,
    *,
    random_crop: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply identical spatial crop coordinates to every frame."""
    if video.ndim != 4:
        raise ValueError(f"Expected [T,C,H,W], got {tuple(video.shape)}")
    height, width = video.shape[-2:]
    if crop_size > height or crop_size > width:
        raise ValueError(
            f"crop_size={crop_size} exceeds resized frame {(height, width)}"
        )
    if random_crop:
        top = int(torch.randint(height - crop_size + 1, (), generator=generator))
        left = int(torch.randint(width - crop_size + 1, (), generator=generator))
    else:
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2
    return video[..., top : top + crop_size, left : left + crop_size]


def gnvc_train_preprocess(
    video: torch.Tensor,
    *,
    resize_to: int = 512,
    crop_size: int = 256,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    video = resize_short_side(video, resize_to)
    return shared_crop(video, crop_size, random_crop=True, generator=generator)
