from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from entropy_model.model_categorical_context_corruption import (
    CategoricalEntropyModelContextCorruption,
)


def temporal_radius_attention_mask(
    num_frames: int,
    num_groups: int,
    sites_per_block: int,
    radius: int | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_blocks = num_frames * num_groups
    query_block = torch.arange(num_blocks, device=device)[:, None]
    key_block = torch.arange(num_blocks, device=device)[None, :]
    allowed = key_block <= query_block
    if radius is not None:
        if radius <= 0:
            raise ValueError(f"temporal radius must be positive, got {radius}")
        target_frame = query_block // num_groups
        source_frame = (key_block - 1).clamp_min(0) // num_groups
        bos = key_block == 0
        same_frame_source = source_frame == target_frame
        recent_history = source_frame >= target_frame - radius
        allowed &= bos | same_frame_source | recent_history
    token_allowed = allowed.repeat_interleave(
        sites_per_block, dim=0
    ).repeat_interleave(sites_per_block, dim=1)
    mask = torch.zeros(token_allowed.shape, device=device, dtype=dtype)
    mask.masked_fill_(~token_allowed, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)


def logits_with_temporal_radius(
    model: CategoricalEntropyModelContextCorruption,
    selected: torch.Tensor,
    radius: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = model.layout.to_blocks(selected.permute(0, 2, 3, 4, 1))
    hidden = model._shifted_inputs(blocks.values).reshape(
        blocks.values.shape[0],
        blocks.values.shape[1] * model.layout.sites_per_block,
        model.config.d_model,
    )
    mask = temporal_radius_attention_mask(
        blocks.latent_frames,
        model.layout.num_groups,
        model.layout.sites_per_block,
        radius,
        device=hidden.device,
        dtype=hidden.dtype,
    )
    for layer in model.transformer.layers:
        hidden = layer(hidden, mask)
    hidden = hidden.reshape(
        blocks.values.shape[0],
        blocks.values.shape[1],
        model.layout.sites_per_block,
        model.config.d_model,
    )
    logits = model.output_head(model.output_norm(hidden))
    return logits, mask[0, 0] == 0


def evaluate_temporal_radii(
    model: CategoricalEntropyModelContextCorruption,
    selected: torch.Tensor,
    indices: torch.Tensor,
    radii: list[int],
    rate_pixels: int,
    device: torch.device,
) -> list[dict]:
    targets = model._targets(indices)
    dense_predictions: torch.Tensor | None = None
    dense_allowed = 0
    reports: list[dict] = []
    for radius in [None, *radii]:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        logits, token_allowed = logits_with_temporal_radius(model, selected, radius)
        torch.cuda.synchronize(device)
        forward_latency_ms = (time.perf_counter() - started) * 1000.0
        bits = F.cross_entropy(
            logits.float().reshape(-1, model.config.codebook_size),
            targets.reshape(-1),
            reduction="none",
        ) / math.log(2.0)
        predictions = logits.argmax(dim=-1)
        if dense_predictions is None:
            dense_predictions = predictions
            dense_allowed = int(token_allowed.sum())
        reports.append(
            {
                "temporal_radius": "all" if radius is None else radius,
                "bits_per_index": float(bits.mean().cpu()),
                "estimated_bpp": float(bits.sum().cpu()) / rate_pixels,
                "top1_accuracy": float((predictions == targets).float().mean().cpu()),
                "dense_top1_agreement": float(
                    (predictions == dense_predictions).float().mean().cpu()
                ),
                "logical_attention_density": int(token_allowed.sum()) / dense_allowed,
                "dense_kernel_forward_latency_ms": forward_latency_ms,
            }
        )
    dense_bpi = reports[0]["bits_per_index"]
    dense_bpp = reports[0]["estimated_bpp"]
    for report in reports:
        report["delta_bits_per_index"] = report["bits_per_index"] - dense_bpi
        report["delta_bpp"] = report["estimated_bpp"] - dense_bpp
    return reports
