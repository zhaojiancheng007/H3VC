from __future__ import annotations

from dataclasses import dataclass
import math
import shutil
from pathlib import Path
import time
from typing import Any

from DISTS_pytorch import DISTS
import numpy as np
from PIL import Image
import torch

from datasets.video_sequence import load_png_sequence
from entropy_model.model_categorical import CategoricalEntropyModel
from metrics.lpips import sequence_lpips
from tokenizer.stage1_vq import Stage1VQTokenizer
from tokenizer.stage3_vq import selected_code_vectors
from tokenizer.stage_backbones import TOKENIZER_SPECS
from training.evaluation_categorical import (
    DATASET_NAMES,
    HEVC_B_SEQUENCES,
    MCL_JCV_SEQUENCES,
    UVG_SEQUENCES,
    _estimated_bits,
    _sequence_dists,
)
from training.uvg_protocol import (
    finish_uvg_reconstruction,
    prepare_uvg_input,
    uvg_metric_frames,
    uvg_source_frames,
)


def codebook_statistics(counts: torch.Tensor) -> dict[str, Any]:
    total = int(counts.sum().item())
    if total == 0:
        raise ValueError("Cannot measure codebook utilization without assignments")
    probabilities = counts[counts > 0].double() / total
    entropy_bits = float(-(probabilities * probabilities.log2()).sum().item())
    return {
        "total_assignments": total,
        "codebook_size": counts.numel(),
        "active_codes": int((counts > 0).sum().item()),
        "marginal_entropy_bits": entropy_bits,
        "perplexity": 2.0 ** entropy_bits,
        "top1_mass": float(probabilities.max().item()),
    }


@dataclass(frozen=True)
class SequenceJob:
    dataset: str
    sequence: str
    frames_dir: Path


def save_reconstruction(reconstruction: torch.Tensor, output_dir: Path) -> None:
    output_dir.mkdir(parents=True)
    frames = reconstruction.mul(255.0).round().to(torch.uint8)
    for frame_index, frame in enumerate(frames):
        pixels = frame.permute(1, 2, 0).numpy()
        Image.fromarray(np.ascontiguousarray(pixels)).save(
            output_dir / f"{frame_index + 1:06d}.png"
        )


def replace_reconstruction_archive(
    source_dir: Path, canonical_dir: Path
) -> None:
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    if canonical_dir.exists():
        shutil.rmtree(canonical_dir)
    source_dir.replace(canonical_dir)


def build_benchmark_jobs(
    uvg_root: str | Path,
    mcl_jcv_root: str | Path,
    hevc_b_root: str | Path,
) -> tuple[SequenceJob, ...]:
    roots = {
        "uvg": Path(uvg_root),
        "mcl_jcv": Path(mcl_jcv_root),
        "hevc_b": Path(hevc_b_root),
    }
    jobs = [
        SequenceJob(
            "uvg",
            name,
            roots["uvg"] / f"{name}_1920x1080_120fps_420_8bit_YUV_frames",
        )
        for name in UVG_SEQUENCES
    ]
    jobs.extend(
        SequenceJob("mcl_jcv", name, roots["mcl_jcv"] / name)
        for name in MCL_JCV_SEQUENCES
    )
    jobs.extend(
        SequenceJob("hevc_b", name, roots["hevc_b"] / name)
        for name in HEVC_B_SEQUENCES
    )
    return tuple(jobs)


def shard_jobs(
    jobs: tuple[SequenceJob, ...], rank: int, world_size: int
) -> tuple[SequenceJob, ...]:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0,{world_size}), got {rank}")
    return jobs[rank::world_size]


@torch.inference_mode()
def evaluate_sequence_job(
    job: SequenceJob,
    tokenizer: Stage1VQTokenizer,
    entropy_model: CategoricalEntropyModel | None,
    tokenizer_name: str,
    lpips_vgg_metric: torch.nn.Module,
    lpips_alex_metric: torch.nn.Module,
    dists_metric: DISTS,
    device: torch.device,
    recon_root: Path | None,
    *,
    k: int | None,
) -> dict[str, Any]:
    source_frames = uvg_source_frames(tokenizer_name)
    video, _ = load_png_sequence(job.frames_dir, source_frames, start_frame=1)
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
        projected.shape[0],
        projected.shape[2],
        projected.shape[3],
        projected.shape[4],
    )
    selected = selected_code_vectors(tokenizer.quantizer.embed, indices)
    if entropy_model is None:
        bits_per_index = math.log2(tokenizer.quantizer.embed.shape[0])
        bits = float(indices.numel() * bits_per_index)
    else:
        if k is None:
            raise ValueError("Entropy evaluation requires an explicit K")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bits = float(
                _estimated_bits(entropy_model, selected, indices, k=k).item()
            )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        reconstruction = tokenizer.decode(
            tokenizer.post_quant(selected.to(projected.dtype))
        )
    torch.cuda.synchronize(device)
    total_ms = (time.perf_counter() - started) * 1000.0
    counts = torch.bincount(flat_indices, minlength=tokenizer.quantizer.embed.shape[0]).cpu()

    reconstruction = finish_uvg_reconstruction(reconstruction, prepared)
    if recon_root is not None:
        save_reconstruction(reconstruction, recon_root / job.dataset / job.sequence)
    reference = prepared.reference
    difference = reference - reconstruction
    squared_error = float(difference.square().sum().item())
    value_count = difference.numel()
    rate_frames = prepared.rate_frames
    rate_pixels = rate_frames * reference.shape[-2] * reference.shape[-1]
    lpips_vgg = sequence_lpips(
        reference, reconstruction, lpips_vgg_metric, device=device, batch_size=1
    )
    lpips_alex = sequence_lpips(
        reference, reconstruction, lpips_alex_metric, device=device, batch_size=1
    )
    dists = _sequence_dists(reference, reconstruction, dists_metric, device)
    return {
        "codebook": codebook_statistics(counts),
        "codebook_histogram": counts.tolist(),
        "dataset": job.dataset,
        "sequence": job.sequence,
        "frames": reference.shape[0],
        "rate_frames": rate_frames,
        "resolution": [reference.shape[-2], reference.shape[-1]],
        "psnr": -10.0 * math.log10(squared_error / value_count),
        "bpp": bits / rate_pixels,
        "lpips": lpips_vgg,
        "lpips_vgg": lpips_vgg,
        "lpips_alex": lpips_alex,
        "dists": dists,
        "latency_ms": total_ms / reference.shape[0],
        "bits": bits,
        "squared_error": squared_error,
        "value_count": value_count,
    }


def evaluate_shard(
    jobs: tuple[SequenceJob, ...],
    tokenizer: Stage1VQTokenizer,
    entropy_model: CategoricalEntropyModel | None,
    tokenizer_name: str,
    lpips_vgg_metric: torch.nn.Module,
    lpips_alex_metric: torch.nn.Module,
    dists_metric: DISTS,
    device: torch.device,
    recon_root: Path | None,
    *,
    k: int | None,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    reports = [
        evaluate_sequence_job(
            job,
            tokenizer,
            entropy_model,
            tokenizer_name,
            lpips_vgg_metric,
            lpips_alex_metric,
            dists_metric,
            device,
            recon_root,
            k=k,
        )
        for job in shard_jobs(jobs, rank, world_size)
    ]
    return {
        "schema_version": 1,
        "rank": rank,
        "world_size": world_size,
        "tokenizer": tokenizer_name,
        "protocol": {
            "model_input_frames": TOKENIZER_SPECS[tokenizer_name].input_frames,
            "metric_frames": uvg_metric_frames(tokenizer_name),
            "bpp_denominator_frames": uvg_source_frames(tokenizer_name),
            "rate": (
                f"{math.log2(tokenizer.quantizer.embed.shape[0]):g} raw bits "
                "per index without entropy coding"
                if entropy_model is None
                else "estimated categorical cross-entropy bits"
            ),
            "entropy_k": k,
        },
        "sequences": reports,
    }


def _expected_sequence_names(dataset: str) -> tuple[str, ...]:
    return {
        "uvg": UVG_SEQUENCES,
        "mcl_jcv": MCL_JCV_SEQUENCES,
        "hevc_b": HEVC_B_SEQUENCES,
    }[dataset]


def _aggregate_dataset(
    dataset: str,
    reports: list[dict[str, Any]],
    tokenizer_name: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    total_frames = sum(int(report["frames"]) for report in reports)
    total_rate_pixels = sum(
        int(report["rate_frames"])
        * int(report["resolution"][0])
        * int(report["resolution"][1])
        for report in reports
    )
    total_error = sum(float(report["squared_error"]) for report in reports)
    total_values = sum(int(report["value_count"]) for report in reports)
    sequences = [
        {name: value for name, value in report.items() if name != "dataset"}
        for report in reports
    ]
    return {
        "schema_version": 2,
        "dataset": DATASET_NAMES[dataset],
        "tokenizer": tokenizer_name,
        "protocol": {
            "sequences": list(_expected_sequence_names(dataset)),
            **protocol,
        },
        "aggregate": {
            "psnr": -10.0 * math.log10(total_error / total_values),
            "bpp": sum(float(report["bits"]) for report in reports)
            / total_rate_pixels,
            "lpips": sum(
                float(report["lpips"]) * int(report["frames"])
                for report in reports
            )
            / total_frames,
            "lpips_vgg": sum(
                float(report["lpips_vgg"]) * int(report["frames"])
                for report in reports
            )
            / total_frames,
            "lpips_alex": sum(
                float(report["lpips_alex"]) * int(report["frames"])
                for report in reports
            )
            / total_frames,
            "dists": sum(
                float(report["dists"]) * int(report["frames"])
                for report in reports
            )
            / total_frames,
            "latency_ms": sum(
                float(report["latency_ms"]) * int(report["frames"])
                for report in reports
            )
            / total_frames,
        },
        "sequences": sequences,
    }


def merge_evaluation_shards(
    shards: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    world_sizes = {int(shard["world_size"]) for shard in shards}
    if len(world_sizes) != 1:
        raise ValueError(f"Evaluation shards disagree on world_size: {world_sizes}")
    world_size = world_sizes.pop()
    ranks = {int(shard["rank"]) for shard in shards}
    if ranks != set(range(world_size)):
        raise ValueError(
            f"Expected shard ranks 0..{world_size - 1}, got {sorted(ranks)}"
        )
    tokenizers = {str(shard["tokenizer"]) for shard in shards}
    if len(tokenizers) != 1:
        raise ValueError(f"Evaluation shards disagree on tokenizer: {tokenizers}")
    tokenizer_name = tokenizers.pop()
    checkpoint_steps = {int(shard["checkpoint_step"]) for shard in shards}
    if len(checkpoint_steps) != 1:
        raise ValueError(
            f"Evaluation shards disagree on checkpoint step: {checkpoint_steps}"
        )
    checkpoint_step = checkpoint_steps.pop()
    protocol = shards[0]["protocol"]
    if any(shard["protocol"] != protocol for shard in shards[1:]):
        raise ValueError("Evaluation shards disagree on protocol")

    by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for shard in shards:
        for report in shard["sequences"]:
            identity = (str(report["dataset"]), str(report["sequence"]))
            if identity in by_identity:
                raise ValueError(f"Duplicate sequence evaluation: {identity}")
            by_identity[identity] = report
    expected = {
        (dataset, sequence)
        for dataset in ("uvg", "mcl_jcv", "hevc_b")
        for sequence in _expected_sequence_names(dataset)
    }
    if set(by_identity) != expected:
        missing = sorted(expected - set(by_identity))
        extra = sorted(set(by_identity) - expected)
        raise ValueError(
            f"Evaluation shard coverage changed: missing={missing}, extra={extra}"
        )

    outputs = {}
    for dataset in ("uvg", "mcl_jcv", "hevc_b"):
        reports = [
            by_identity[(dataset, sequence)]
            for sequence in _expected_sequence_names(dataset)
        ]
        outputs[dataset] = _aggregate_dataset(
            dataset, reports, tokenizer_name, protocol
        )
        outputs[dataset]["checkpoint_step"] = checkpoint_step
    return outputs
