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
from training.evaluation_progressive_groups_distributed import (
    PROGRESSIVE_LEVELS,
    merge_progressive_shards,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run clean dynamic-rank progressive Stage2 evaluation"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu-ids", required=True, nargs="+", type=int)
    parser.add_argument("--uvg-root", default="/luojix/data/UVG")
    parser.add_argument(
        "--mcl-jcv-root", default="/luojix/data/MCL-JCV/figure4_bt709_first96"
    )
    parser.add_argument(
        "--hevc-b-root", default="/luojix/data/HEVC-B/figure4_bt709_first96"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=False)
    world_size = len(args.gpu_ids)
    processes = []
    for rank, gpu_id in enumerate(args.gpu_ids):
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/evaluate_stage2_dynamic_rank_clean_progressive_distributed.py"),
            "--checkpoint", str(args.checkpoint.resolve()),
            "--config", str(args.config.resolve()),
            "--output", str(shards_dir / f"rank_{rank:02d}.json"),
            "--rank", str(rank),
            "--world-size", str(world_size),
            "--uvg-root", str(Path(args.uvg_root).resolve()),
            "--mcl-jcv-root", str(Path(args.mcl_jcv_root).resolve()),
            "--hevc-b-root", str(Path(args.hevc_b_root).resolve()),
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
        raise RuntimeError(f"Progressive eval workers failed: {failures}")

    shards = [
        json.loads((shards_dir / f"rank_{rank:02d}.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]
    outputs = merge_progressive_shards(shards)
    summary = {
        "schema_version": 1,
        "experiment": "stage2_dynamic_rank_clean_progressive_groups",
        "checkpoint": str(args.checkpoint.resolve()),
        "world_size": world_size,
        "gpu_ids": args.gpu_ids,
        "levels": {},
    }
    for level in PROGRESSIVE_LEVELS:
        summary["levels"][str(level)] = {}
        for dataset, report in outputs[level].items():
            write_json(output_dir / f"level_{level}" / f"{dataset}.json", report)
            summary["levels"][str(level)][dataset] = report["aggregate"]
    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
