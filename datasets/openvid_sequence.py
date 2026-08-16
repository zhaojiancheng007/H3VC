from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .transforms import gnvc_train_preprocess
from .video_sequence import load_png_sequence


class OpenVidTrainingDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        dataset_root: str | Path,
        metadata_path: str | Path,
        *,
        num_frames: int = 33,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.sequence_root = self.dataset_root / "sequences"
        self.metadata_path = Path(metadata_path)
        self.num_frames = num_frames
        if num_frames != 33:
            raise ValueError(f"OpenVid AR training requires 33 frames, got {num_frames}")
        if not self.sequence_root.is_dir():
            raise FileNotFoundError(f"Missing OpenVid sequence root: {self.sequence_root}")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Missing OpenVid metadata: {self.metadata_path}")

        self.records: list[dict] = []
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                if int(record.get("frames", -1)) != num_frames:
                    raise ValueError(
                        f"{self.metadata_path}:{line_number} does not describe {num_frames} frames"
                    )
                sequence = record.get("sequence")
                if not isinstance(sequence, str) or not sequence:
                    raise ValueError(f"{self.metadata_path}:{line_number} has no sequence id")
                sequence_dir = self.sequence_root / sequence
                if not sequence_dir.is_dir():
                    raise FileNotFoundError(
                        f"Metadata sequence {sequence} has no directory: {sequence_dir}"
                    )
                self.records.append(record)
        if not self.records:
            raise ValueError(f"No OpenVid records found in {self.metadata_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        sequence = self.records[index]["sequence"]
        video, _ = load_png_sequence(self.sequence_root / sequence, self.num_frames)
        crop_seed = int(torch.randint(0, 2**31 - 1, ()).item())
        generator = torch.Generator().manual_seed(crop_seed)
        cropped = gnvc_train_preprocess(video, generator=generator)
        return cropped.float().div(127.5).sub(1.0).permute(1, 0, 2, 3)
