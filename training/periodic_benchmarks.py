from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


BENCHMARKS_BY_RANK = (
    ("uvg", "uvg_root"),
    ("mcl_jcv", "mcl_jcv_root"),
    ("hevc_b", "hevc_b_root"),
)


def benchmark_for_rank(
    args: argparse.Namespace,
    rank: int,
    evaluation_dir: Path,
) -> tuple[str, Path, Path] | None:
    if rank >= len(BENCHMARKS_BY_RANK):
        return None
    dataset, root_argument = BENCHMARKS_BY_RANK[rank]
    return dataset, Path(getattr(args, root_argument)), evaluation_dir / f"{dataset}.json"


def evaluation_environment(local_rank: int) -> dict[str, str]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical_device = (
        str(local_rank)
        if visible_devices is None
        else visible_devices.split(",")[local_rank]
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = physical_device
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)
    return environment


def evaluation_command(
    args: argparse.Namespace,
    stage: int,
    checkpoint_path: Path,
    dataset: str,
    dataset_root: Path,
    output_path: Path,
    project_root: Path,
) -> list[str]:
    evaluator = {
        1: "evaluate_stage1_tokenizer_uvg.py",
        2: "evaluate_stage2_entropy_uvg.py",
        3: "evaluate_stage3_joint_uvg_v2.py",
    }[stage]
    command = [
        sys.executable,
        str(project_root / "scripts" / evaluator),
        "--checkpoint",
        str(checkpoint_path.resolve()),
        "--dataset",
        dataset,
        "--dataset-root",
        str(dataset_root.resolve()),
        "--output",
        str(output_path.resolve()),
    ]
    if stage in (2, 3):
        command.extend(("--config", str(Path(args.config).resolve())))
    command.extend(
        (
            "--wan-repo",
            args.wan_repo,
            "--wan-checkpoint",
            args.wan_checkpoint,
            "--cosmos-repo",
            args.cosmos_repo,
            "--cosmos-encoder",
            args.cosmos_encoder,
            "--cosmos-decoder",
            args.cosmos_decoder,
            "--cosmos-cv888-encoder",
            args.cosmos_cv888_encoder,
            "--cosmos-cv888-decoder",
            args.cosmos_cv888_decoder,
            "--minimax-h3-model-dir",
            args.minimax_h3_model_dir,
        )
    )
    return command
