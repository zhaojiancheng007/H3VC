from __future__ import annotations

import argparse
from pathlib import Path

from training.periodic_benchmarks_gaussian import distributed_evaluation_command


def dynamic_rank_clean_evaluation_command(
    args: argparse.Namespace,
    stage: int,
    checkpoint_path: Path,
    evaluation_dir: Path,
    rank: int,
    world_size: int,
    project_root: Path,
) -> list[str]:
    if stage != 2:
        raise ValueError(f"Clean dynamic-rank evaluator only supports Stage2, got {stage}")
    command = distributed_evaluation_command(
        args,
        stage,
        checkpoint_path,
        evaluation_dir,
        rank,
        world_size,
        project_root,
    )
    command[1] = str(
        project_root / "scripts" / "evaluate_stage2_dynamic_rank_clean_distributed.py"
    )
    return command
