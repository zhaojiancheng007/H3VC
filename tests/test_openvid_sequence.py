import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from datasets.openvid_sequence import OpenVidTrainingDataset
from datasets.openvid_sequence_ddp import (
    SeededOpenVidTrainingDataset,
    ThreadedOpenVidBatchDataset,
    TrainingSampleKey,
)
from datasets.openvid_preresized_sequence import (
    CACHE_MANIFEST_NAME,
    INCOMPLETE_MARKER_NAME,
    SeededPreResizedOpenVidTrainingDataset,
    file_sha256,
)


def test_openvid_pngs_are_loaded_and_preprocessed_online(tmp_path: Path) -> None:
    sequence_dir = tmp_path / "sequences" / "000001"
    sequence_dir.mkdir(parents=True)
    for frame in range(1, 34):
        pixels = np.full((512, 512, 3), frame, dtype=np.uint8)
        Image.fromarray(pixels).save(sequence_dir / f"{frame:06d}.png")
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        json.dumps({"sequence": "000001", "frames": 33}) + "\n",
        encoding="utf-8",
    )

    dataset = OpenVidTrainingDataset(tmp_path, metadata)
    video = dataset[0]

    assert len(dataset) == 1
    assert video.shape == (3, 33, 256, 256)
    assert video.dtype == torch.float32
    assert float(video.min()) >= -1.0
    assert float(video.max()) <= 1.0
    expected = torch.arange(1, 34, dtype=torch.float32).div(127.5).sub(1.0)
    assert torch.allclose(video[0, :, 0, 0], expected)


def test_seeded_threaded_openvid_batch_matches_sequential_loading(
    tmp_path: Path,
) -> None:
    sequence_dir = tmp_path / "sequences" / "000001"
    sequence_dir.mkdir(parents=True)
    for frame in range(1, 34):
        pixels = np.full((64, 80, 3), frame, dtype=np.uint8)
        Image.fromarray(pixels).save(sequence_dir / f"{frame:06d}.png")
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        json.dumps({"sequence": "000001", "frames": 33}) + "\n",
        encoding="utf-8",
    )

    dataset = SeededOpenVidTrainingDataset(tmp_path, metadata)
    keys = [TrainingSampleKey(0, 41), TrainingSampleKey(0, 73)]
    expected = torch.stack([dataset[key] for key in keys])
    threaded = ThreadedOpenVidBatchDataset(dataset, threads=2)
    actual = threaded[keys]
    threaded.close()

    assert torch.equal(actual, expected)


def _write_preresized_fixture(root: Path) -> Path:
    sequence_dir = root / "sequences" / "000001"
    sequence_dir.mkdir(parents=True)
    row = np.arange(640, dtype=np.uint16)[None, :, None]
    column = np.arange(512, dtype=np.uint16)[:, None, None]
    for frame in range(1, 34):
        pixels = np.broadcast_to((row + column + frame) % 256, (512, 640, 3)).astype(
            np.uint8
        )
        Image.fromarray(pixels).save(sequence_dir / f"{frame:06d}.png")
    metadata = root / "metadata.jsonl"
    metadata.write_text(
        json.dumps({"sequence": "000001", "frames": 33}) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "cache_type": "openvid_short_side_rgb_png",
        "frames_per_sequence": 33,
        "resize_short_side": 512,
        "interpolation": "torch_bicubic_align_corners_false",
        "storage_dtype": "uint8",
        "rounding": "torch_round",
        "color_mode": "RGB",
        "metadata_sha256": file_sha256(metadata),
        "sequence_count": 1,
    }
    (root / CACHE_MANIFEST_NAME).write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    return metadata


def test_preresized_loader_matches_online_path_for_512_short_side(
    tmp_path: Path,
) -> None:
    metadata = _write_preresized_fixture(tmp_path)
    key = TrainingSampleKey(0, 73)

    online = SeededOpenVidTrainingDataset(tmp_path, metadata)[key]
    preresized = SeededPreResizedOpenVidTrainingDataset(tmp_path, metadata)[key]

    assert torch.equal(preresized, online)


def test_preresized_loader_rejects_incomplete_cache(tmp_path: Path) -> None:
    metadata = _write_preresized_fixture(tmp_path)
    (tmp_path / INCOMPLETE_MARKER_NAME).write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="cache is incomplete"):
        SeededPreResizedOpenVidTrainingDataset(tmp_path, metadata)


def test_preresized_loader_rejects_metadata_hash_mismatch(tmp_path: Path) -> None:
    metadata = _write_preresized_fixture(tmp_path)
    metadata.write_text(
        json.dumps({"sequence": "000001", "frames": 33, "changed": True}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metadata hash mismatch"):
        SeededPreResizedOpenVidTrainingDataset(tmp_path, metadata)
