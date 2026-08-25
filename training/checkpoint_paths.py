from __future__ import annotations

from pathlib import Path


def assert_checkpoint_reference(
    recorded: str | Path,
    supplied: str | Path,
    *,
    label: str,
    allow_relocation: bool,
) -> bool:
    recorded_path = Path(recorded).resolve()
    supplied_path = Path(supplied).resolve(strict=True)
    if recorded_path == supplied_path:
        return False
    if not allow_relocation:
        raise ValueError(
            f"{label} changed from {recorded_path} to {supplied_path}; "
            "pass --allow-checkpoint-relocation only when this is an exact "
            "copy relocated between machines"
        )
    return True
