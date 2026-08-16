import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from datasets.openvid_mp4_sequence import (
    SeededOpenVidMp4TrainingDataset,
    TrainingSampleKey,
    load_mp4_window,
)


class FakeVideoCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.position = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def set(self, property_id: int, value: float) -> bool:
        assert property_id == cv2.CAP_PROP_POS_FRAMES
        self.position = int(value)
        return True

    def get(self, property_id: int) -> float:
        assert property_id == cv2.CAP_PROP_POS_FRAMES
        return float(self.position)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame.copy()

    def release(self) -> None:
        self.released = True


def test_load_mp4_window_seeks_and_converts_bgr_to_rgb() -> None:
    frames = [
        np.full((3, 4, 3), (index, index + 1, index + 2), dtype=np.uint8)
        for index in range(8)
    ]
    capture = FakeVideoCapture(frames)

    with patch(
        "datasets.openvid_mp4_sequence.cv2.VideoCapture", return_value=capture
    ):
        video = load_mp4_window("sample.mp4", start_frame=3, num_frames=2)

    assert video.shape == (2, 3, 3, 4)
    assert video.dtype == torch.uint8
    assert torch.equal(video[0, :, 0, 0], torch.tensor([5, 4, 3]))
    assert torch.equal(video[1, :, 0, 0], torch.tensor([6, 5, 4]))
    assert capture.released


def test_load_mp4_window_rejects_invalid_frame_shape() -> None:
    capture = FakeVideoCapture([np.zeros((3, 4), dtype=np.uint8)])

    with patch(
        "datasets.openvid_mp4_sequence.cv2.VideoCapture", return_value=capture
    ), pytest.raises(ValueError, match="Expected BGR frame"):
        load_mp4_window("sample.mp4", start_frame=0, num_frames=1)

    assert capture.released


def _write_dataset(tmp_path: Path, *, member: str = "source.mp4") -> Path:
    video_root = tmp_path / "videos"
    video_root.mkdir()
    (video_root / "source.mp4").touch()
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        json.dumps(
            {
                "available_sequence_lengths": [5, 33],
                "height": 512,
                "member": member,
                "samples": {
                    "5": {
                        "frames": 5,
                        "sample_end_frame_inclusive": 15,
                        "sample_start_frame": 11,
                        "scene_end_frame_exclusive": 40,
                        "scene_start_frame": 0,
                    },
                    "33": {
                        "frames": 33,
                        "sample_end_frame_inclusive": 39,
                        "sample_start_frame": 7,
                        "scene_end_frame_exclusive": 40,
                        "scene_start_frame": 0,
                    },
                },
                "sequence": "000001",
                "width": 640,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata


def test_mp4_dataset_uses_member_and_seeded_shared_crop(tmp_path: Path) -> None:
    metadata = _write_dataset(tmp_path)
    frames = torch.arange(33, dtype=torch.uint8)[:, None, None, None].expand(
        33, 3, 512, 640
    )
    dataset = SeededOpenVidMp4TrainingDataset(tmp_path, metadata)
    key = TrainingSampleKey(index=0, crop_seed=41)

    with patch(
        "datasets.openvid_mp4_sequence.load_mp4_window", return_value=frames
    ) as load:
        first = dataset[key]
        second = dataset[key]

    assert load.call_args_list[0].args == (
        tmp_path / "videos" / "source.mp4",
        7,
        33,
    )
    assert first.shape == (3, 33, 256, 256)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)
    expected = torch.arange(33, dtype=torch.float32).div(127.5).sub(1.0)
    assert torch.allclose(first[0, :, 0, 0], expected)


def test_mp4_dataset_selects_requested_sequence_length(tmp_path: Path) -> None:
    metadata = _write_dataset(tmp_path)
    frames = torch.zeros((5, 3, 512, 640), dtype=torch.uint8)
    dataset = SeededOpenVidMp4TrainingDataset(
        tmp_path, metadata, num_frames=5
    )

    with patch(
        "datasets.openvid_mp4_sequence.load_mp4_window", return_value=frames
    ) as load:
        result = dataset[TrainingSampleKey(index=0, crop_seed=2)]

    assert load.call_args.args == (
        tmp_path / "videos" / "source.mp4",
        11,
        5,
    )
    assert result.shape == (3, 5, 256, 256)


def test_mp4_dataset_rejects_unavailable_sequence_length(tmp_path: Path) -> None:
    metadata = _write_dataset(tmp_path)

    with pytest.raises(ValueError, match="No OpenVid records support 49 frames"):
        SeededOpenVidMp4TrainingDataset(tmp_path, metadata, num_frames=49)


@pytest.mark.parametrize("member", ["../source.mp4", "/source.mp4", "source.png"])
def test_mp4_dataset_rejects_unsafe_member(tmp_path: Path, member: str) -> None:
    metadata = _write_dataset(tmp_path, member=member)

    with pytest.raises(ValueError, match="unsafe MP4 member"):
        SeededOpenVidMp4TrainingDataset(tmp_path, metadata)
