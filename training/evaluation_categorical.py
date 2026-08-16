from __future__ import annotations

import math
from pathlib import Path
import time

from DISTS_pytorch import DISTS
import torch
import torch.nn.functional as F

from datasets.video_sequence import load_png_sequence
from entropy_model.model_categorical import CategoricalEntropyModel
from metrics.lpips import sequence_lpips
from tokenizer.cosmos_dv.preprocessing import pad_for_cosmos, unpad_reconstruction
from tokenizer.stage1_vq import Stage1VQTokenizer
from tokenizer.stage3_vq import selected_code_vectors


UVG_FRAMES = 96
UVG_SEGMENTS = (25, 25, 25, 21)
UVG_SEQUENCES = (
    "Beauty",
    "Bosphorus",
    "HoneyBee",
    "Jockey",
    "ReadySteadyGo",
    "ShakeNDry",
    "YachtRide",
)


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
) -> torch.Tensor:
    window = entropy_model.config.window_size
    pad_height = (-indices.shape[-2]) % window
    pad_width = (-indices.shape[-1]) % window
    valid = torch.ones_like(indices, dtype=torch.bool)
    if pad_height or pad_width:
        features = F.pad(features, (0, pad_width, 0, pad_height))
        indices = F.pad(indices, (0, pad_width, 0, pad_height))
        valid = F.pad(valid, (0, pad_width, 0, pad_height), value=False)
    logits, _ = entropy_model.predict(features)
    targets = entropy_model.layout.to_blocks(indices.unsqueeze(-1)).values.squeeze(-1)
    valid_blocks = entropy_model.layout.to_blocks(valid.unsqueeze(-1)).values.squeeze(-1)
    bits = F.cross_entropy(
        logits.float().reshape(-1, entropy_model.config.codebook_size),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape) / math.log(2.0)
    return bits[valid_blocks].sum()


@torch.inference_mode()
def evaluate_uvg(
    tokenizer: Stage1VQTokenizer,
    entropy_model: CategoricalEntropyModel,
    tokenizer_name: str,
    uvg_root: str | Path,
    lpips_metric: torch.nn.Module,
    dists_metric: DISTS,
    device: torch.device,
) -> dict:
    tokenizer.eval()
    entropy_model.eval()
    reports = []
    for sequence_name in UVG_SEQUENCES:
        frames_dir = (
            Path(uvg_root)
            / f"{sequence_name}_1920x1080_120fps_420_8bit_YUV_frames"
        )
        video, _ = load_png_sequence(frames_dir, UVG_FRAMES, start_frame=1)
        frame_offset = 0
        squared_error = 0.0
        value_count = 0
        estimated_bits = 0.0
        lpips_sum = 0.0
        dists_sum = 0.0
        total_ms = 0.0

        for frame_count in UVG_SEGMENTS:
            reference = video[frame_offset : frame_offset + frame_count].float().div(255.0)
            model_input = (
                reference.permute(1, 0, 2, 3).unsqueeze(0).mul(2.0).sub(1.0)
            )
            pad_info = None
            if tokenizer_name in {"cosmos_cv488", "cosmos_cv888"}:
                temporal_alignment = 4 if tokenizer_name == "cosmos_cv488" else 8
                model_input, pad_info = pad_for_cosmos(
                    model_input,
                    temporal_alignment=temporal_alignment,
                    spatial_alignment=16,
                )
            model_input = model_input.to(device)

            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latent = tokenizer.encode(model_input)
                projected = tokenizer.pre_quant(latent)
            flat = (
                projected.permute(0, 2, 3, 4, 1)
                .reshape(-1, projected.shape[1])
                .float()
            )
            flat_indices = tokenizer.quantizer._nearest_indices(flat)
            indices = flat_indices.reshape(
                projected.shape[0],
                projected.shape[2],
                projected.shape[3],
                projected.shape[4],
            )
            selected = selected_code_vectors(tokenizer.quantizer.embed, indices)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                current_bits = _estimated_bits(entropy_model, selected, indices)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                reconstruction = tokenizer.decode(
                    tokenizer.post_quant(selected.to(projected.dtype))
                )
            torch.cuda.synchronize(device)
            total_ms += (time.perf_counter() - started) * 1000.0

            if pad_info is not None:
                reconstruction = unpad_reconstruction(reconstruction, pad_info)
            reconstruction = (
                reconstruction.float().clamp(-1.0, 1.0).add(1.0).div(2.0)
                .squeeze(0).permute(1, 0, 2, 3).cpu()
            )
            difference = reference - reconstruction
            squared_error += float(difference.square().sum().item())
            value_count += difference.numel()
            estimated_bits += float(current_bits.item())
            lpips_sum += sequence_lpips(
                reference,
                reconstruction,
                lpips_metric,
                device=device,
                batch_size=1,
            ) * frame_count
            dists_sum += _sequence_dists(
                reference, reconstruction, dists_metric, device
            ) * frame_count
            frame_offset += frame_count

        pixels = frame_offset * video.shape[-2] * video.shape[-1]
        reports.append(
            {
                "sequence": sequence_name,
                "frames": frame_offset,
                "psnr": -10.0 * math.log10(squared_error / value_count),
                "bpp": estimated_bits / pixels,
                "lpips": lpips_sum / frame_offset,
                "dists": dists_sum / frame_offset,
                "latency_ms": total_ms / frame_offset,
                "estimated_bits": estimated_bits,
                "squared_error": squared_error,
                "value_count": value_count,
            }
        )

    total_frames = sum(report["frames"] for report in reports)
    total_pixels = total_frames * 1080 * 1920
    total_error = sum(report["squared_error"] for report in reports)
    total_values = sum(report["value_count"] for report in reports)
    return {
        "schema_version": 1,
        "dataset": "UVG",
        "tokenizer": tokenizer_name,
        "protocol": {
            "sequences": list(UVG_SEQUENCES),
            "frames_per_sequence": UVG_FRAMES,
            "segments": list(UVG_SEGMENTS),
            "rate": "estimated 4096-way categorical cross-entropy bits",
        },
        "aggregate": {
            "psnr": -10.0 * math.log10(total_error / total_values),
            "bpp": sum(report["estimated_bits"] for report in reports) / total_pixels,
            "lpips": sum(report["lpips"] * report["frames"] for report in reports)
            / total_frames,
            "dists": sum(report["dists"] * report["frames"] for report in reports)
            / total_frames,
            "latency_ms": sum(
                report["latency_ms"] * report["frames"] for report in reports
            ) / total_frames,
        },
        "sequences": reports,
    }

@torch.inference_mode()
def evaluate_uvg_v2(
    tokenizer: Stage1VQTokenizer,
    entropy_model: CategoricalEntropyModel | None,
    tokenizer_name: str,
    uvg_root: str | Path,
    lpips_metric: torch.nn.Module,
    dists_metric: DISTS,
    device: torch.device,
) -> dict:
    from training.uvg_protocol import (
        finish_uvg_reconstruction,
        prepare_uvg_input,
        uvg_metric_frames,
        uvg_source_frames,
    )

    tokenizer.eval()
    if entropy_model is None:
        bits_per_index = math.log2(tokenizer.quantizer.embed.shape[0])
        rate_description = (
            f"{bits_per_index:g} raw bits per index without entropy coding"
        )
    else:
        entropy_model.eval()
        rate_description = "estimated categorical cross-entropy bits"
    source_frames = uvg_source_frames(tokenizer_name)
    reports = []
    for sequence_name in UVG_SEQUENCES:
        frames_dir = (
            Path(uvg_root)
            / f"{sequence_name}_1920x1080_120fps_420_8bit_YUV_frames"
        )
        video, _ = load_png_sequence(frames_dir, source_frames, start_frame=1)
        prepared = prepare_uvg_input(video.float().div(255.0), tokenizer_name)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent = tokenizer.encode(prepared.model_input.to(device))
            projected = tokenizer.pre_quant(latent)
        flat = projected.permute(0, 2, 3, 4, 1).reshape(
            -1, projected.shape[1]
        ).float()
        flat_indices = tokenizer.quantizer._nearest_indices(flat)
        indices = flat_indices.reshape(
            projected.shape[0], projected.shape[2],
            projected.shape[3], projected.shape[4],
        )
        selected = selected_code_vectors(tokenizer.quantizer.embed, indices)
        if entropy_model is None:
            bits = indices.numel() * bits_per_index
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                bits = float(_estimated_bits(entropy_model, selected, indices).item())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reconstruction = tokenizer.decode(
                tokenizer.post_quant(selected.to(projected.dtype))
            )
        torch.cuda.synchronize(device)
        total_ms = (time.perf_counter() - started) * 1000.0
        reconstruction = finish_uvg_reconstruction(reconstruction, prepared)
        reference = prepared.reference
        difference = reference - reconstruction
        squared_error = float(difference.square().sum().item())
        value_count = difference.numel()
        rate_frames = prepared.model_frames
        rate_pixels = rate_frames * reference.shape[-2] * reference.shape[-1]
        reports.append({
            "sequence": sequence_name,
            "frames": reference.shape[0],
            "rate_frames": rate_frames,
            "resolution": [reference.shape[-2], reference.shape[-1]],
            "psnr": -10.0 * math.log10(squared_error / value_count),
            "bpp": bits / rate_pixels,
            "lpips": sequence_lpips(
                reference, reconstruction, lpips_metric, device=device, batch_size=1
            ),
            "dists": _sequence_dists(reference, reconstruction, dists_metric, device),
            "latency_ms": total_ms / reference.shape[0],
            "bits": bits,
            "squared_error": squared_error,
            "value_count": value_count,
        })

    total_frames = sum(report["frames"] for report in reports)
    total_rate_pixels = sum(
        report["rate_frames"]
        * report["resolution"][0]
        * report["resolution"][1]
        for report in reports
    )
    total_error = sum(report["squared_error"] for report in reports)
    total_values = sum(report["value_count"] for report in reports)
    return {
        "schema_version": 2,
        "dataset": "UVG",
        "tokenizer": tokenizer_name,
        "protocol": {
            "sequences": list(UVG_SEQUENCES),
            "model_input_frames": source_frames,
            "metric_frames": uvg_metric_frames(tokenizer_name),
            "bpp_denominator_frames": source_frames,
            "rate": rate_description,
        },
        "aggregate": {
            "psnr": -10.0 * math.log10(total_error / total_values),
            "bpp": sum(report["bits"] for report in reports)
            / total_rate_pixels,
            "lpips": sum(report["lpips"] * report["frames"] for report in reports)
            / total_frames,
            "dists": sum(report["dists"] * report["frames"] for report in reports)
            / total_frames,
            "latency_ms": sum(
                report["latency_ms"] * report["frames"] for report in reports
            ) / total_frames,
        },
        "sequences": reports,
    }
