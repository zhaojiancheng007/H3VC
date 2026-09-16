from __future__ import annotations

import math

from DISTS_pytorch import DISTS
import torch
import torch.nn.functional as F

from entropy_model.model_categorical import CategoricalEntropyModel


UVG_SEQUENCES = (
    "Beauty",
    "Bosphorus",
    "HoneyBee",
    "Jockey",
    "ReadySteadyGo",
    "ShakeNDry",
    "YachtRide",
)
MCL_JCV_SEQUENCES = tuple(f"videoSRC{index:02d}" for index in range(1, 31))
HEVC_B_SEQUENCES = (
    "BasketballDrive_1920x1080_50",
    "BQTerrace_1920x1080_60",
    "Cactus_1920x1080_50",
    "Kimono1_1920x1080_24",
    "ParkScene_1920x1080_24",
)
DATASET_NAMES = {
    "uvg": "UVG",
    "mcl_jcv": "MCL-JCV",
    "hevc_b": "HEVC-B",
}


def _sequence_dists(
    reference: torch.Tensor,
    reconstruction: torch.Tensor,
    metric: DISTS,
    device: torch.device,
) -> float:
    total = 0.0
    for frame in range(reference.shape[0]):
        value = metric(
            reference[frame : frame + 1].to(device),
            reconstruction[frame : frame + 1].to(device),
            require_grad=False,
            batch_average=False,
        )
        total += float(value.item())
    return total / reference.shape[0]


def _estimated_bits(
    entropy_model: CategoricalEntropyModel,
    features: torch.Tensor,
    indices: torch.Tensor,
    *,
    k: int,
) -> torch.Tensor:
    window = entropy_model.config.window_size
    pad_height = (-indices.shape[-2]) % window
    pad_width = (-indices.shape[-1]) % window
    valid = torch.ones_like(indices, dtype=torch.bool)
    if pad_height or pad_width:
        features = F.pad(features, (0, pad_width, 0, pad_height))
        indices = F.pad(indices, (0, pad_width, 0, pad_height))
        valid = F.pad(valid, (0, pad_width, 0, pad_height), value=False)
    logits, _ = entropy_model.predict(features, k=k)
    layout = entropy_model.layout(k)
    targets = layout.to_blocks(indices.unsqueeze(-1)).values.squeeze(-1)
    valid_blocks = layout.to_blocks(valid.unsqueeze(-1)).values.squeeze(-1)
    bits = F.cross_entropy(
        logits.float().reshape(-1, entropy_model.config.codebook_size),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape) / math.log(2.0)
    return bits[valid_blocks].sum()


__all__ = [
    "DATASET_NAMES",
    "HEVC_B_SEQUENCES",
    "MCL_JCV_SEQUENCES",
    "UVG_SEQUENCES",
    "_estimated_bits",
    "_sequence_dists",
]
