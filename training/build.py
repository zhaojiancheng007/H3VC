from __future__ import annotations

import torch

from datasets.transforms import gnvc_train_preprocess


def build_training_sample(video_uint8: torch.Tensor, seed: int) -> torch.Tensor:
    """Core data path for the future entropy-model training stage."""
    if video_uint8.shape[0] != 33:
        raise ValueError(f"Training sample must contain 33 frames, got {video_uint8.shape[0]}")
    generator = torch.Generator().manual_seed(seed)
    cropped = gnvc_train_preprocess(video_uint8, generator=generator)
    return cropped.float().div(127.5).sub(1.0).permute(1, 0, 2, 3)
