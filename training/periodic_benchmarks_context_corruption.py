from __future__ import annotations

import argparse
from pathlib import Path

from training.periodic_benchmarks_gaussian import distributed_evaluation_command


def context_corruption_evaluation_command(
    args: argparse.Namespace,
    stage: int,
    checkpoint_path: Path,
    evaluation_dir: Path,
    rank: int,
    world_size: int,
    project_root: Path,
) -> list[str]:
    if stage != 2:
        raise ValueError(f"Context-corruption evaluator only supports Stage2, got {stage}")
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
        project_root / "scripts" / "evaluate_stage2_context_corruption_distributed.py"
    )
    return command
