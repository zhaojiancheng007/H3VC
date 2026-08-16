import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from datasets.openvid_mp4_sequence import TrainingSampleKey
from datasets.trainable_jsons_sequence import SeededTrainableJsonsVideoDataset


def _write_metadata(tmp_path: Path, lengths: tuple[int, ...]) -> tuple[Path, Path]:
    dataset_root = tmp_path / "ddn" / "gemini"
    video_dir = dataset_root / "gemini-sharedata" / "videos"
    video_dir.mkdir(parents=True)
    metadata_dir = tmp_path / "trainable_jsons_v1"
    metadata_dir.mkdir()
    clips = []
    for index, length in enumerate(lengths):
        video_path = video_dir / f"{index}.mp4"
        video_path.touch()
        clips.append(
            {
                "video_path": f"/workspace/gemini/gemini-sharedata/videos/{index}.mp4",
                "canny_path": "",
                "video_length": length,
                "video_width": 640,
                "video_height": 512,
                "fps": 25,
                "prompt_path": "",
            }
        )
    (metadata_dir / "valid_all_chunk_1.json").write_text(
        json.dumps({"clips": clips, "origin": "/workspace/gemini/source"}),
        encoding="utf-8",
    )
    return dataset_root, metadata_dir


def test_dataset_filters_short_clips_and_replays_seeded_sample(tmp_path: Path) -> None:
    dataset_root, metadata_dir = _write_metadata(tmp_path, (33, 24))
    dataset = SeededTrainableJsonsVideoDataset(
        dataset_root, metadata_dir, num_frames=25
    )
    frames = torch.arange(25, dtype=torch.uint8)[:, None, None, None].expand(
        25, 3, 512, 640
    )
    key = TrainingSampleKey(index=0, crop_seed=41)
    generator = torch.Generator().manual_seed(key.crop_seed)
    expected_start = int(torch.randint(9, (), generator=generator))

    with patch(
        "datasets.trainable_jsons_sequence.load_mp4_window", return_value=frames
    ) as load:
        first = dataset[key]
        second = dataset[key]

    assert len(dataset) == 1
    assert load.call_args_list[0].args == (
        dataset_root / "gemini-sharedata" / "videos" / "0.mp4",
        expected_start,
        25,
    )
    assert first.shape == (3, 25, 256, 256)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)


def test_dataset_rejects_collection_without_long_enough_clip(tmp_path: Path) -> None:
    dataset_root, metadata_dir = _write_metadata(tmp_path, (3, 24))

    with pytest.raises(ValueError, match="No records support 25 frames"):
        SeededTrainableJsonsVideoDataset(dataset_root, metadata_dir, num_frames=25)


def test_dataset_deterministically_replaces_decode_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root, metadata_dir = _write_metadata(tmp_path, (33, 33))
    dataset = SeededTrainableJsonsVideoDataset(
        dataset_root, metadata_dir, num_frames=25
    )
    rejection_log = tmp_path / "rejected_samples.jsonl"
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("ARCODEC_REJECTED_SAMPLES_LOG", str(rejection_log))
    frames = torch.zeros((25, 3, 512, 640), dtype=torch.uint8)
    key = TrainingSampleKey(index=0, crop_seed=41)
    generator = torch.Generator().manual_seed(key.crop_seed)
    first_start = int(torch.randint(9, (), generator=generator))
    replacement_rank = int(torch.randint(1, (), generator=generator))
    replacement_index = replacement_rank + 1
    replacement_start = int(torch.randint(9, (), generator=generator))

    with patch(
        "datasets.trainable_jsons_sequence.load_mp4_window",
        side_effect=[RuntimeError("decode failed"), frames],
    ) as load:
        video = dataset[key]

    assert load.call_args_list[0].args == (
        dataset_root / "gemini-sharedata" / "videos" / "0.mp4",
        first_start,
        25,
    )
    assert load.call_args_list[1].args == (
        dataset_root / "gemini-sharedata" / "videos" / f"{replacement_index}.mp4",
        replacement_start,
        25,
    )
    assert video.shape == (3, 25, 256, 256)
    record = json.loads(rejection_log.read_text(encoding="utf-8"))
    assert record["rank"] == 3
    assert record["rejected_index"] == 0
    assert record["replacement_index"] == replacement_index
    assert record["error"] == "decode failed"
