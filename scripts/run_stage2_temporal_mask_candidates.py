from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.reporting import write_json
from training.temporal_attention_mask_candidates import CANDIDATE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run temporal causal mask diagnostics across selected GPUs"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu-ids", required=True, nargs="+", type=int)
    parser.add_argument("--sample-indices", required=True, nargs="+", type=int)
    parser.add_argument("--crop-seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.sample_indices) < len(args.gpu_ids):
        raise ValueError("sample count must be at least the GPU count")
    if len(set(args.sample_indices)) != len(args.sample_indices):
        raise ValueError("sample indices must be unique")
    output_dir = args.output_dir.resolve()
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=False)
    world_size = len(args.gpu_ids)
    processes = []
    for rank, gpu_id in enumerate(args.gpu_ids):
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/analyze_stage2_temporal_mask_candidates_distributed.py"),
            "--checkpoint", str(args.checkpoint.resolve()),
            "--config", str(args.config.resolve()),
            "--output", str(shards_dir / f"rank_{rank:02d}.json"),
            "--rank", str(rank),
            "--world-size", str(world_size),
            "--sample-indices", *[str(index) for index in args.sample_indices],
            "--crop-seed", str(args.crop_seed),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            environment.pop(name, None)
        log_path = output_dir / f"rank_{rank:02d}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, log_handle, log_path))

    failures = []
    for process, log_handle, log_path in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code != 0:
            failures.append((return_code, log_path))
    if failures:
        raise RuntimeError(f"Temporal mask workers failed: {failures}")

    shards = [
        json.loads((shards_dir / f"rank_{rank:02d}.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]
    samples = [sample for shard in shards for sample in shard["samples"]]
    if sorted(sample["sample_index"] for sample in samples) != sorted(args.sample_indices):
        raise ValueError("Temporal mask shard coverage changed")
    aggregate = {}
    for candidate in CANDIDATE_NAMES:
        reports = [
            next(item for item in sample["candidates"] if item["candidate"] == candidate)
            for sample in samples
        ]
        bits = sum(report["bits"] for report in reports)
        indices = sum(report["indices"] for report in reports)
        rate_pixels = sum(report["rate_pixels"] for report in reports)
        aggregate[candidate] = {
            "bits_per_index": bits / indices,
            "estimated_bpp": bits / rate_pixels,
            "top1_accuracy": sum(report["top1_correct"] for report in reports) / indices,
            "dense_top1_agreement": sum(
                report["dense_top1_agree"] for report in reports
            ) / indices,
            "logical_attention_density": sum(
                report["logical_allowed"] for report in reports
            ) / sum(report["dense_logical_allowed"] for report in reports),
            "latency_ms": sum(report["latency_ms"] for report in reports) / len(reports),
        }
    dense = aggregate["dense"]
    for report in aggregate.values():
        report["delta_bits_per_index"] = report["bits_per_index"] - dense["bits_per_index"]
        report["delta_bpp"] = report["estimated_bpp"] - dense["estimated_bpp"]
    write_json(
        output_dir / "report.json",
        {
            "schema_version": 1,
            "experiment": "stage2_temporal_causal_mask_candidates",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_step": int(shards[0]["checkpoint_step"]),
            "sample_indices": sorted(args.sample_indices),
            "crop_seed": args.crop_seed,
            "world_size": world_size,
            "gpu_ids": args.gpu_ids,
            "protocol": {
                "spatial_windows": "all 8x8 windows stay parallel",
                "causality": "keys are restricted to BOS and shifted source blocks available before each target block",
                "anchors": "all four groups from historical frames at temporal lags divisible by 5",
                "latency": "dense PyTorch SDPA intervention; not sparse-kernel speed",
            },
            "layer_specs": shards[0]["layer_specs"],
            "aggregate": aggregate,
            "samples": sorted(samples, key=lambda sample: sample["sample_index"]),
        },
    )


if __name__ == "__main__":
    main()
