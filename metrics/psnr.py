from __future__ import annotations

import math

import torch


def sequence_psnr(reference: torch.Tensor, reconstruction: torch.Tensor) -> float:
    if reference.shape != reconstruction.shape:
        raise ValueError(f"PSNR shape mismatch: {reference.shape} vs {reconstruction.shape}")
    mse = torch.mean((reference.float() - reconstruction.float()) ** 2).item()
    if mse == 0:
        return math.inf
    return -10.0 * math.log10(mse)
