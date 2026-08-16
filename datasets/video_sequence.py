from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def list_numbered_pngs(frames_dir: str | Path, num_frames: int, start_frame: int = 1) -> list[Path]:
    frames_dir = Path(frames_dir)
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frame directory does not exist: {frames_dir}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")

    paths = sorted(frames_dir.glob("*.png"))
    by_number: dict[int, Path] = {}
    for path in paths:
        if not path.stem.isdecimal():
            raise ValueError(f"PNG filename must have a numeric stem: {path.name}")
        number = int(path.stem)
        if number in by_number:
            raise ValueError(f"Duplicate frame number {number} in {frames_dir}")
        by_number[number] = path

    expected = list(range(start_frame, start_frame + num_frames))
    missing = [number for number in expected if number not in by_number]
    if missing:
        raise ValueError(
            f"Missing required consecutive frames in {frames_dir}: {missing[:8]}"
        )
    return [by_number[number] for number in expected]


def load_png_sequence(
    frames_dir: str | Path,
    num_frames: int,
    start_frame: int = 1,
) -> tuple[torch.Tensor, list[Path]]:
    """Load consecutive RGB PNGs as uint8 [T,C,H,W]."""
    paths = list_numbered_pngs(frames_dir, num_frames, start_frame)
    frames: list[torch.Tensor] = []
    size: tuple[int, int] | None = None
    for path in paths:
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise ValueError(f"Expected RGB PNG, got mode={image.mode!r}: {path}")
            array = np.asarray(image)
        current_size = (int(array.shape[0]), int(array.shape[1]))
        if size is None:
            size = current_size
        elif current_size != size:
            raise ValueError(
                f"Frame size changed from {size} to {current_size}: {path}"
            )
        frames.append(torch.from_numpy(array.copy()).permute(2, 0, 1))
    return torch.stack(frames, dim=0), paths
