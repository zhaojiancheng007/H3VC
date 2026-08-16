from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from datasets.video_sequence import load_png_sequence


def _write_rgb(path: Path, value: int) -> None:
    Image.fromarray(np.full((8, 12, 3), value, dtype=np.uint8)).save(path)


def test_load_consecutive_rgb_sequence(tmp_path: Path) -> None:
    for index in range(1, 4):
        _write_rgb(tmp_path / f"{index:06d}.png", index)
    video, paths = load_png_sequence(tmp_path, 3)
    assert video.shape == (3, 3, 8, 12)
    assert [path.name for path in paths] == ["000001.png", "000002.png", "000003.png"]


def test_missing_frame_fails(tmp_path: Path) -> None:
    _write_rgb(tmp_path / "000001.png", 1)
    _write_rgb(tmp_path / "000003.png", 3)
    with pytest.raises(ValueError, match="Missing required consecutive"):
        load_png_sequence(tmp_path, 3)
