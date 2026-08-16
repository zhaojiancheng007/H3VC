from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint


def create_lpips_vgg(device: torch.device) -> torch.nn.Module:
    import lpips

    metric = lpips.LPIPS(
        net="vgg",
        version="0.1",
        lpips=True,
        spatial=False,
        verbose=False,
    ).to(device).eval()
    metric.requires_grad_(False)
    return metric


def video_lpips(
    reference: torch.Tensor,
    reconstruction: torch.Tensor,
    metric: torch.nn.Module,
) -> torch.Tensor:
    if reference.shape != reconstruction.shape:
        raise ValueError(
            f"LPIPS video shape mismatch: {tuple(reference.shape)} vs "
            f"{tuple(reconstruction.shape)}"
        )
    reference_frames = reference.permute(0, 2, 1, 3, 4).reshape(
        -1, 3, reference.shape[-2], reference.shape[-1]
    )
    reconstruction_frames = reconstruction.permute(0, 2, 1, 3, 4).reshape_as(
        reference_frames
    )
    values = metric(
        reference_frames.float(), reconstruction_frames.float(), normalize=False
    )
    return values.float().mean()


def chunked_video_lpips(
    reference: torch.Tensor,
    reconstruction: torch.Tensor,
    metric: torch.nn.Module,
    *,
    frame_batch_size: int,
    activation_checkpointing: bool,
) -> torch.Tensor:
    if reference.shape != reconstruction.shape:
        raise ValueError(
            f"LPIPS video shape mismatch: {tuple(reference.shape)} vs "
            f"{tuple(reconstruction.shape)}"
        )
    if frame_batch_size <= 0:
        raise ValueError("frame_batch_size must be positive")
    reference_frames = reference.permute(0, 2, 1, 3, 4).reshape(
        -1, 3, reference.shape[-2], reference.shape[-1]
    )
    reconstruction_frames = reconstruction.permute(0, 2, 1, 3, 4).reshape_as(
        reference_frames
    )
    frame_count = reference_frames.shape[0]
    total = reference_frames.new_zeros((), dtype=torch.float32)
    for start in range(0, frame_count, frame_batch_size):
        end = min(start + frame_batch_size, frame_count)
        reference_batch = reference_frames[start:end].float()
        reconstruction_batch = reconstruction_frames[start:end].float()
        if activation_checkpointing:
            values = checkpoint(
                lambda ref, rec: metric(ref, rec, normalize=False),
                reference_batch,
                reconstruction_batch,
                use_reentrant=False,
            )
        else:
            values = metric(reference_batch, reconstruction_batch, normalize=False)
        total = total + values.sum()
    return total / frame_count
