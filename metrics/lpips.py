from __future__ import annotations

import torch
from torch import nn


def create_standard_lpips(device: str | torch.device) -> nn.Module:
    import lpips

    return lpips.LPIPS(
        net="alex",
        version="0.1",
        lpips=True,
        spatial=False,
        verbose=False,
    ).to(device).eval()


@torch.inference_mode()
def sequence_lpips(
    reference: torch.Tensor,
    reconstruction: torch.Tensor,
    metric: nn.Module,
    *,
    device: str | torch.device,
    batch_size: int = 1,
) -> float:
    if reference.shape != reconstruction.shape:
        raise ValueError(
            f"LPIPS shape mismatch: {reference.shape} vs {reconstruction.shape}"
        )
    if reference.ndim != 4 or reference.shape[1] != 3:
        raise ValueError(f"Expected RGB video [T,3,H,W], got {reference.shape}")
    if reference.shape[0] <= 0:
        raise ValueError("LPIPS requires at least one frame")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    for name, value in (("reference", reference), ("reconstruction", reconstruction)):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} contains non-finite values")
        minimum = float(value.amin().item())
        maximum = float(value.amax().item())
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(
                f"{name} must be in [0,1], got min={minimum}, max={maximum}"
            )

    total = 0.0
    frame_count = int(reference.shape[0])
    for start in range(0, frame_count, batch_size):
        end = min(start + batch_size, frame_count)
        reference_batch = (
            reference[start:end].to(device=device, dtype=torch.float32).mul(2.0).sub(1.0)
        )
        reconstruction_batch = (
            reconstruction[start:end]
            .to(device=device, dtype=torch.float32)
            .mul(2.0)
            .sub(1.0)
        )
        values = metric(reference_batch, reconstruction_batch, normalize=False)
        if not isinstance(values, torch.Tensor) or values.numel() != end - start:
            raise RuntimeError(
                "LPIPS must return exactly one scalar per frame, got "
                f"{type(values).__name__} with shape "
                f"{getattr(values, 'shape', None)}"
            )
        if not torch.isfinite(values).all():
            raise FloatingPointError("LPIPS returned non-finite values")
        total += float(values.double().sum().item())
    return total / frame_count
