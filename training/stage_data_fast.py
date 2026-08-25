from __future__ import annotations

from pathlib import Path

from datasets.trainable_jsons_sequence_fast import (
    SeededFastTrainableJsonsVideoDataset,
)


DEFAULT_DATASET_FORMAT = "trainable_jsons"
DEFAULT_DATASET_ROOT = "/ddn/gemini"
DEFAULT_METADATA = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "trainable_jsons_v1_filtered_decode_failures_step0"
)


def build_stage_dataset_fast(
    dataset_format: str,
    dataset_root: str | Path,
    metadata: str | Path,
    *,
    num_frames: int,
    output_dir: str | Path,
):
    if dataset_format != "trainable_jsons":
        raise ValueError(
            "Fast loader supports only dataset_format='trainable_jsons', "
            f"got {dataset_format!r}"
        )
    return SeededFastTrainableJsonsVideoDataset(
        dataset_root,
        metadata,
        num_frames=num_frames,
        rejection_log=Path(output_dir) / "rejected_samples.jsonl",
    )
