#!/usr/bin/env python3
"""Plot progressive zero-shot points in the standard ARCodec RD setting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from plot_rd_curves import (
    METRICS,
    baseline_records,
    manifest_records,
    plot,
    read_json,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_SUMMARY = (
    ROOT.parent
    / "results"
    / "minimax_h3_vq4096x24_s1_55k_s2_55k_20260822"
    / "stage3"
    / "lambda_7p2_tau_0p05"
    / "progressive_groups"
    / "step_0015000"
    / "summary.json"
)
DEFAULT_KNN_SUMMARY = (
    ROOT.parent
    / "results"
    / "entropy_model_prefix_coding"
    / "context_corruption_knn8_p025_step_0025000"
    / "summary.json"
)
DEFAULT_BASELINE = ROOT / "baseline_rd_points.json"
DEFAULT_MANIFEST = ROOT.parent.parent / "eval_archive" / "manifest.json"
DATASETS = (("uvg", "UVG"), ("mcl_jcv", "MCL-JCV"))
LEVELS = (1, 2, 3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--knn-summary", type=Path, default=DEFAULT_KNN_SUMMARY)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    return parser.parse_args()


def read_progressive_records(
    path: Path, method: str = "ARCodec progressive zero-shot"
) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported progressive summary schema: {payload.get('schema_version')}"
        )
    if set(map(int, payload["levels"])) != set(LEVELS):
        raise ValueError(f"Expected progressive levels {LEVELS}")

    required = {
        "bpp",
        "psnr",
        "lpips_vgg",
        "lpips_alex",
        "dists",
        "latency_ms",
    }
    records = []
    for level in LEVELS:
        for dataset_key, dataset_name in DATASETS:
            aggregate = payload["levels"][str(level)][dataset_key]
            missing = required - set(aggregate)
            if missing:
                raise ValueError(
                    f"Level {level} {dataset_key} is missing metrics: {sorted(missing)}"
                )
            for metric_name, metric_key in METRICS:
                records.append(
                    {
                        "dataset": dataset_name,
                        "metric": metric_name,
                        "method": method,
                        "label": f"K={level}",
                        "bpp": float(aggregate["bpp"]),
                        "value": float(aggregate[metric_key]),
                    }
                )
    return records


def main() -> None:
    args = parse_args()
    datasets, archived_baseline, _ = manifest_records(read_json(args.manifest))
    baseline = baseline_records(read_json(args.baseline)) + archived_baseline
    build_dir = args.output_dir / "zeroshot_rd_build"
    plot(
        datasets,
        baseline,
        read_progressive_records(args.summary)
        + read_progressive_records(
            args.knn_summary,
            method="ARCodec progressive zero-shot + KNN 25k",
        ),
        build_dir,
        metrics=METRICS,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "rd_curves.png").replace(
        args.output_dir / "zeroshot_rd_curves.png"
    )
    (build_dir / "rd_curves.pdf").replace(
        args.output_dir / "zeroshot_rd_curves.pdf"
    )
    build_dir.rmdir()
    print(args.output_dir / "zeroshot_rd_curves.png")
    print(args.output_dir / "zeroshot_rd_curves.pdf")


if __name__ == "__main__":
    main()
