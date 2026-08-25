from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from entropy_model.model_categorical_context_corruption import (
    CategoricalEntropyModelContextCorruption,
)


CANDIDATE_NAMES = (
    "dense",
    "radius_8",
    "radius_16",
    "radius_4_anchor_5",
    "radius_8_anchor_5",
    "layerwise_anchor_5",
)


def temporal_hybrid_attention_mask(
    num_frames: int,
    num_groups: int,
    sites_per_block: int,
    *,
    radius: int | None,
    anchor_stride: int | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if radius is not None and radius <= 0:
        raise ValueError(f"temporal radius must be positive, got {radius}")
    if anchor_stride is not None and anchor_stride <= 0:
        raise ValueError(f"anchor stride must be positive, got {anchor_stride}")
    if radius is None and anchor_stride is not None:
        raise ValueError("anchor_stride requires a finite temporal radius")

    num_blocks = num_frames * num_groups
    query_block = torch.arange(num_blocks, device=device)[:, None]
    key_block = torch.arange(num_blocks, device=device)[None, :]
    allowed = key_block <= query_block
    if radius is not None:
        target_frame = query_block // num_groups
        source_frame = (key_block - 1).clamp_min(0) // num_groups
        temporal_lag = target_frame - source_frame
        bos = key_block == 0
        same_frame_source = source_frame == target_frame
        recent_history = (temporal_lag > 0) & (temporal_lag <= radius)
        if anchor_stride is None:
            anchor_history = torch.zeros_like(recent_history)
        else:
            anchor_history = (temporal_lag > 0) & (
                temporal_lag.remainder(anchor_stride) == 0
            )
        allowed &= bos | same_frame_source | recent_history | anchor_history

    token_allowed = allowed.repeat_interleave(
        sites_per_block, dim=0
    ).repeat_interleave(sites_per_block, dim=1)
    mask = torch.zeros(token_allowed.shape, device=device, dtype=dtype)
    mask.masked_fill_(~token_allowed, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)


def candidate_layer_specs(name: str, num_layers: int) -> list[tuple[int | None, int | None]]:
    if name == "dense":
        return [(None, None)] * num_layers
    if name == "radius_8":
        return [(8, None)] * num_layers
    if name == "radius_16":
        return [(16, None)] * num_layers
    if name == "radius_4_anchor_5":
        return [(4, 5)] * num_layers
    if name == "radius_8_anchor_5":
        return [(8, 5)] * num_layers
    if name == "layerwise_anchor_5":
        if num_layers != 8:
            raise ValueError(
                f"layerwise_anchor_5 is defined for 8 layers, got {num_layers}"
            )
        return [
            (None, None),
            (8, 5),
            (8, 5),
            (4, 5),
            (4, 5),
            (8, 5),
            (8, 5),
            (16, 5),
        ]
    raise ValueError(f"Unknown temporal mask candidate: {name}")


def logits_with_temporal_mask_candidate(
    model: CategoricalEntropyModelContextCorruption,
    selected: torch.Tensor,
    candidate: str,
) -> tuple[torch.Tensor, int, int]:
    blocks = model.layout.to_blocks(selected.permute(0, 2, 3, 4, 1))
    hidden = model._shifted_inputs(blocks.values).reshape(
        blocks.values.shape[0],
        blocks.values.shape[1] * model.layout.sites_per_block,
        model.config.d_model,
    )
    allowed_count = 0
    dense_allowed_count = 0
    for layer, (radius, anchor_stride) in zip(
        model.transformer.layers,
        candidate_layer_specs(candidate, len(model.transformer.layers)),
        strict=True,
    ):
        mask = temporal_hybrid_attention_mask(
            blocks.latent_frames,
            model.layout.num_groups,
            model.layout.sites_per_block,
            radius=radius,
            anchor_stride=anchor_stride,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        allowed_count += int((mask == 0).sum())
        num_blocks = blocks.values.shape[1]
        sites = model.layout.sites_per_block
        dense_allowed_count += (
            num_blocks * (num_blocks + 1) // 2 * sites * sites
        )
        hidden = layer(hidden, mask)
    hidden = hidden.reshape(
        blocks.values.shape[0],
        blocks.values.shape[1],
        model.layout.sites_per_block,
        model.config.d_model,
    )
    logits = model.output_head(model.output_norm(hidden))
    return logits, allowed_count, dense_allowed_count


def evaluate_temporal_mask_candidates(
    model: CategoricalEntropyModelContextCorruption,
    selected: torch.Tensor,
    indices: torch.Tensor,
    rate_pixels: int,
    device: torch.device,
) -> list[dict]:
    targets = model._targets(indices)
    dense_predictions: torch.Tensor | None = None
    reports = []
    for candidate in CANDIDATE_NAMES:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        logits, allowed_count, dense_allowed_count = logits_with_temporal_mask_candidate(
            model, selected, candidate
        )
        torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        bits = F.cross_entropy(
            logits.float().reshape(-1, model.config.codebook_size),
            targets.reshape(-1),
            reduction="none",
        ) / math.log(2.0)
        predictions = logits.argmax(dim=-1)
        if dense_predictions is None:
            dense_predictions = predictions
        reports.append(
            {
                "candidate": candidate,
                "bits": float(bits.sum().cpu()),
                "indices": bits.numel(),
                "rate_pixels": rate_pixels,
                "bits_per_index": float(bits.mean().cpu()),
                "estimated_bpp": float(bits.sum().cpu()) / rate_pixels,
                "top1_correct": int((predictions == targets).sum().cpu()),
                "dense_top1_agree": int((predictions == dense_predictions).sum().cpu()),
                "logical_allowed": allowed_count,
                "dense_logical_allowed": dense_allowed_count,
                "logical_attention_density": allowed_count / dense_allowed_count,
                "latency_ms": latency_ms,
            }
        )
    dense_bits_per_index = reports[0]["bits_per_index"]
    dense_bpp = reports[0]["estimated_bpp"]
    for report in reports:
        report["delta_bits_per_index"] = (
            report["bits_per_index"] - dense_bits_per_index
        )
        report["delta_bpp"] = report["estimated_bpp"] - dense_bpp
    return reports
