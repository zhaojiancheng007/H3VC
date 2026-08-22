from __future__ import annotations

import argparse
import sys
from pathlib import Path

DISTRIBUTED_GAUSSIAN_EVALUATORS = {
    1: "evaluate_stage1_tokenizer_gaussian_distributed.py",
    2: "evaluate_stage2_cosine_gaussian_distributed.py",
    3: "evaluate_stage3_cosine_gaussian_distributed.py",
}


def distributed_evaluation_command(
    args: argparse.Namespace,
    stage: int,
    checkpoint_path: Path,
    evaluation_dir: Path,
    rank: int,
    world_size: int,
    project_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(project_root / "scripts" / DISTRIBUTED_GAUSSIAN_EVALUATORS[stage]),
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
