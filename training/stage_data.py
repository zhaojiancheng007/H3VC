from __future__ import annotations

import os
from pathlib import Path

from datasets.openvid_mp4_sequence import SeededOpenVidMp4TrainingDataset
from datasets.trainable_jsons_sequence import SeededTrainableJsonsVideoDataset


DEFAULT_DATASET_FORMAT = "trainable_jsons"
DEFAULT_DATASET_ROOT = "/ddn/gemini"
DEFAULT_METADATA = (
    "/ddn/gemini/gemini-sharedata/space/qdrnbvyr7uio/yifq/yanjq/"
    "video_data/trainable_jsons_v1"
)


def build_stage_dataset(
    dataset_format: str,
    dataset_root: str | Path,
    metadata: str | Path,
    *,
    num_frames: int,
    output_dir: str | Path,
):
    if dataset_format == "openvid":
        return SeededOpenVidMp4TrainingDataset(
            dataset_root, metadata, num_frames=num_frames
        )
    if dataset_format == "trainable_jsons":
        os.environ["ARCODEC_REJECTED_SAMPLES_LOG"] = str(
            Path(output_dir) / "rejected_samples.jsonl"
        )
        return SeededTrainableJsonsVideoDataset(
            dataset_root, metadata, num_frames=num_frames
        )
    raise ValueError(f"Unknown dataset format {dataset_format!r}")
