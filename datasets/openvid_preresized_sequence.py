from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .openvid_sequence_ddp import SeededOpenVidTrainingDataset, TrainingSampleKey
from .transforms import shared_crop
from .video_sequence import load_png_sequence


CACHE_SCHEMA_VERSION = 1
CACHE_MANIFEST_NAME = "preprocess_manifest.json"
INCOMPLETE_MARKER_NAME = ".preprocess_incomplete.json"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_preresized_manifest(
    dataset_root: str | Path,
    metadata_path: str | Path,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    metadata_path = Path(metadata_path)
    incomplete = dataset_root / INCOMPLETE_MARKER_NAME
    if incomplete.exists():
        raise RuntimeError(f"Pre-resized PNG cache is incomplete: {incomplete}")

    manifest_path = dataset_root / CACHE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing pre-resized PNG manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise TypeError(f"Cache manifest must be a JSON object: {manifest_path}")

    required = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_type": "openvid_short_side_rgb_png",
        "frames_per_sequence": 33,
        "resize_short_side": 512,
        "interpolation": "torch_bicubic_align_corners_false",
        "storage_dtype": "uint8",
        "rounding": "torch_round",
        "color_mode": "RGB",
    }
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in required.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f"Incompatible pre-resized PNG manifest {manifest_path}: {mismatches}"
        )

    expected_metadata_hash = manifest.get("metadata_sha256")
    if not isinstance(expected_metadata_hash, str) or len(expected_metadata_hash) != 64:
        raise ValueError(f"Invalid metadata_sha256 in {manifest_path}")
    actual_metadata_hash = file_sha256(metadata_path)
    if actual_metadata_hash != expected_metadata_hash:
        raise ValueError(
            "Pre-resized metadata hash mismatch: "
            f"manifest={expected_metadata_hash}, actual={actual_metadata_hash}"
        )
    return manifest


class SeededPreResizedOpenVidTrainingDataset(SeededOpenVidTrainingDataset):
    def __init__(
        self,
        dataset_root: str | Path,
        metadata_path: str | Path,
    ) -> None:
        super().__init__(dataset_root, metadata_path)
        self.preprocess_manifest = load_preresized_manifest(
            dataset_root, metadata_path
        )
        manifest_sequences = self.preprocess_manifest.get("sequence_count")
        if manifest_sequences != len(self):
            raise ValueError(
                "Pre-resized sequence count mismatch: "
                f"manifest={manifest_sequences}, metadata={len(self)}"
            )

    def __getitem__(self, key: TrainingSampleKey) -> torch.Tensor:
        if not isinstance(key, TrainingSampleKey):
            raise TypeError(f"Expected TrainingSampleKey, got {type(key).__name__}")
        if key.index < 0 or key.index >= len(self):
            raise IndexError(key.index)
        if key.crop_seed < 0:
            raise ValueError(f"crop_seed must be non-negative, got {key.crop_seed}")

        sequence = self.records[key.index]["sequence"]
        video, _ = load_png_sequence(
            self.sequence_root / sequence, self.num_frames
        )
        height, width = map(int, video.shape[-2:])
        if min(height, width) != 512:
            raise ValueError(
                f"Pre-resized sequence {sequence} has shape {(height, width)}; "
                "expected short side 512"
            )
        generator = torch.Generator().manual_seed(key.crop_seed)
        cropped = shared_crop(
            video, 256, random_crop=True, generator=generator
        )
        return cropped.float().div(127.5).sub(1.0).permute(1, 0, 2, 3)
