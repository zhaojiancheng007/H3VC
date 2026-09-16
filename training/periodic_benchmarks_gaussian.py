from __future__ import annotations

import argparse
import sys
from pathlib import Path


def distributed_evaluation_command(
    args: argparse.Namespace,
    stage: int,
    checkpoint_path: Path,
    evaluation_dir: Path,
    rank: int,
    world_size: int,
    project_root: Path,
) -> list[str]:
    if stage not in {1, 2, 3}:
        raise ValueError(f"stage must be 1, 2, or 3, got {stage}")
    command = [
        sys.executable,
        str(project_root / "scripts" / f"stage{stage}.py"),
        "eval-shard",
        "--checkpoint",
        str(checkpoint_path.resolve()),
        "--output",
        str((evaluation_dir / "shards" / f"rank_{rank:02d}.json").resolve()),
        "--rank",
        str(rank),
        "--world-size",
        str(world_size),
        "--uvg-root",
        str(Path(args.uvg_root).resolve()),
        "--mcl-jcv-root",
        str(Path(args.mcl_jcv_root).resolve()),
        "--hevc-b-root",
        str(Path(args.hevc_b_root).resolve()),
    ]
    if stage == 3:
        command.extend(
            ("--recon-dir", str((evaluation_dir / "recon").resolve()))
        )
    if stage in {2, 3}:
        command.extend(
            (
                "--config",
                str(Path(args.config).resolve()),
                "--k",
                str(args.eval_k),
            )
        )
    command.extend(
        (
            "--minimax-h3-model-dir",
            args.minimax_h3_model_dir,
        )
    )
    return command
