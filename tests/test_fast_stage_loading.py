import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from datasets.openvid_mp4_sequence import TrainingSampleKey
from datasets.trainable_jsons_sequence_fast import (
    SeededFastTrainableJsonsVideoDataset,
    VideoDecodeError,
    load_mp4_window_resize_crop,
)
from training.checkpoint_config_fast import assert_fast_loader_resume_config
from training.distributed_runtime_fast import build_video_dataloader_fast
from training.stage_data_fast import build_stage_dataset_fast


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


def _write_metadata(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset_root = tmp_path / "gemini"
    video_path = dataset_root / "videos" / "sample.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.touch()
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    payload = {
        "clips": [
            {
                "video_path": "/workspace/gemini/videos/sample.mp4",
                "video_length": 30,
                "video_width": 640,
                "video_height": 512,
                "fps": 25,
            }
        ]
    }
    (metadata_dir / "valid_all_chunk_1.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return dataset_root, metadata_dir, video_path


def test_fast_decode_seeks_resizes_crops_and_converts_rgb() -> None:
    frames = [
        np.full((512, 640, 3), (index, index + 1, index + 2), dtype=np.uint8)
        for index in range(8)
    ]
    capture = FakeVideoCapture(frames)
    generator = torch.Generator().manual_seed(41)
    with patch(
        "datasets.trainable_jsons_sequence_fast.cv2.VideoCapture",
        return_value=capture,
    ):
        video = load_mp4_window_resize_crop(
            "sample.mp4", 3, 2, (512, 640), generator
        )

    assert video.shape == (2, 3, 256, 256)
    assert video.dtype == torch.uint8
    assert torch.equal(video[0, :, 0, 0], torch.tensor([5, 4, 3]))
    assert torch.equal(video[1, :, 0, 0], torch.tensor([6, 5, 4]))
    assert capture.released


def test_fast_dataset_replays_seed_and_fails_decode_directly(tmp_path: Path) -> None:
    dataset_root, metadata_dir, video_path = _write_metadata(tmp_path)
    dataset = SeededFastTrainableJsonsVideoDataset(
        dataset_root, metadata_dir, num_frames=25
    )
    key = TrainingSampleKey(index=0, crop_seed=41)
    generator = torch.Generator().manual_seed(key.crop_seed)
    expected_start = int(torch.randint(6, (), generator=generator))
    decoded = torch.zeros((25, 3, 256, 256), dtype=torch.uint8)

    with patch(
        "datasets.trainable_jsons_sequence_fast.load_mp4_window_resize_crop",
        return_value=decoded,
    ) as load:
        first = dataset[key]
        second = dataset[key]
    assert load.call_args_list[0].args[:4] == (
        video_path,
        expected_start,
        25,
        (512, 640),
    )
    assert first.shape == (3, 25, 256, 256)
    assert torch.equal(first, second)

    with patch(
        "datasets.trainable_jsons_sequence_fast.load_mp4_window_resize_crop",
        side_effect=VideoDecodeError("decode failed"),
    ), pytest.raises(VideoDecodeError, match="decode failed"):
        dataset[key]


def test_fast_dataset_deterministically_replaces_decode_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root, metadata_dir, _ = _write_metadata(tmp_path)
    metadata_file = metadata_dir / "valid_all_chunk_1.json"
    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    replacement_path = dataset_root / "videos" / "replacement.mp4"
    replacement_path.touch()
    payload["clips"].append(
        {
            **payload["clips"][0],
            "video_path": "/workspace/gemini/videos/replacement.mp4",
        }
    )
    metadata_file.write_text(json.dumps(payload), encoding="utf-8")
    rejection_log = tmp_path / "rejected_samples.jsonl"
    dataset = SeededFastTrainableJsonsVideoDataset(
        dataset_root,
        metadata_dir,
        num_frames=25,
        rejection_log=rejection_log,
    )
    monkeypatch.setenv("RANK", "3")
    decoded = torch.zeros((25, 3, 256, 256), dtype=torch.uint8)
    key = TrainingSampleKey(index=0, crop_seed=41)

    with patch(
        "datasets.trainable_jsons_sequence_fast.load_mp4_window_resize_crop",
        side_effect=[VideoDecodeError("decode failed"), decoded],
    ) as load:
        video = dataset[key]

    assert load.call_args_list[0].args[0] != load.call_args_list[1].args[0]
    assert video.shape == (3, 25, 256, 256)
    record = json.loads(rejection_log.read_text(encoding="utf-8"))
    assert record["rank"] == 3
    assert record["rejected_index"] == 0
    assert record["replacement_index"] == 1
    assert record["error"] == "decode failed"


def test_fast_dataset_does_not_hide_non_decode_runtime_error(tmp_path: Path) -> None:
    dataset_root, metadata_dir, _ = _write_metadata(tmp_path)
    dataset = SeededFastTrainableJsonsVideoDataset(
        dataset_root,
        metadata_dir,
        num_frames=25,
        rejection_log=tmp_path / "rejected_samples.jsonl",
    )

    with patch(
        "datasets.trainable_jsons_sequence_fast.load_mp4_window_resize_crop",
        side_effect=RuntimeError("unexpected failure"),
    ), pytest.raises(RuntimeError, match="unexpected failure"):
        dataset[TrainingSampleKey(index=0, crop_seed=41)]


def test_fast_stage_builder_enables_rejection_log(tmp_path: Path) -> None:
    dataset_root, metadata_dir, _ = _write_metadata(tmp_path)
    output_dir = tmp_path / "output"
    dataset = build_stage_dataset_fast(
        "trainable_jsons",
        dataset_root,
        metadata_dir,
        num_frames=25,
        output_dir=output_dir,
    )

    assert dataset.rejection_log == output_dir / "rejected_samples.jsonl"


def test_fast_dataloader_requires_benchmarked_worker_count() -> None:
    with pytest.raises(ValueError, match="num_workers=4"):
        build_video_dataloader_fast(
            torch.utils.data.TensorDataset(torch.zeros(1)),
            object(),
            torch.Generator(),
            num_workers=8,
        )


def test_fast_resume_accepts_only_worker_count_change() -> None:
    checkpoint = {
        "model": {"width": 384},
        "optimizer": {"learning_rate": 5e-5},
        "train": {"batch_size": 2, "num_workers": 8, "seed": 2026},
    }
    current = {
        "model": {"width": 384},
        "optimizer": {"learning_rate": 5e-5},
        "train": {"batch_size": 2, "num_workers": 4, "seed": 2026},
    }

    assert_fast_loader_resume_config(checkpoint, current, stage="Stage2")
    assert checkpoint["train"]["num_workers"] == 8
    assert current["train"]["num_workers"] == 4


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("model", "width", 512),
        ("optimizer", "learning_rate", 1e-4),
        ("train", "batch_size", 4),
        ("train", "seed", 2027),
    ],
)
def test_fast_resume_rejects_scientific_config_changes(
    section: str, field: str, value: object
) -> None:
    checkpoint = {
        "model": {"width": 384},
        "optimizer": {"learning_rate": 5e-5},
        "train": {"batch_size": 2, "num_workers": 8, "seed": 2026},
    }
    current = deepcopy(checkpoint)
    current["train"]["num_workers"] = 4
    current[section][field] = value

    with pytest.raises(ValueError, match="scientific config changed"):
        assert_fast_loader_resume_config(checkpoint, current, stage="Stage2")
