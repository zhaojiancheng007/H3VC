from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import gnvc_train_preprocess


def load_consecutive_rgb_frames(
    video_path: str,
    start_frame: int,
    num_frames: int,
) -> torch.Tensor:
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    frames: list[torch.Tensor] = []
    try:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame)):
            raise RuntimeError(f"OpenCV could not seek {path} to frame {start_frame}")
        actual_position = capture.get(cv2.CAP_PROP_POS_FRAMES)
        if abs(actual_position - start_frame) > 0.5:
            raise RuntimeError(
                f"OpenCV seek mismatch for {path}: "
                f"requested={start_frame}, actual={actual_position}"
            )
        frame_size: tuple[int, int] | None = None
        for offset in range(num_frames):
            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise RuntimeError(
                    f"OpenCV stopped at source frame {start_frame + offset}: {path}"
                )
            if bgr.ndim != 3 or bgr.shape[2] != 3:
                raise ValueError(f"Expected a BGR frame, got {bgr.shape}: {path}")
            current_size = (int(bgr.shape[0]), int(bgr.shape[1]))
            if frame_size is None:
                frame_size = current_size
            elif current_size != frame_size:
                raise ValueError(
                    f"Frame size changed from {frame_size} to {current_size}: {path}"
                )
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            frames.append(torch.from_numpy(rgb).permute(2, 0, 1))
    finally:
        capture.release()
    return torch.stack(frames, dim=0)


class SeededInfinityStarMp4TrainingDataset(Dataset[torch.Tensor]):
    def __init__(self, dataset_root: str | Path, *, num_frames: int = 25) -> None:
        self.dataset_root = Path(dataset_root)
        self.metadata_path = self.dataset_root / "raw.jsonl"
        self.num_frames = num_frames
        if num_frames <= 0 or (num_frames - 1) % 4:
            raise ValueError(f"Sequence length must be a positive 4k+1 value, got {num_frames}")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(self.metadata_path)

        self.records: list[tuple[str, int, int]] = []
        self.total_records = 0
        self.short_records = 0
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"{self.metadata_path}:{line_number} is blank")
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError(f"{self.metadata_path}:{line_number} is not an object")
                video_path = record.get("video_path")
                begin = record.get("begin_frame_id")
                end = record.get("end_frame_id")
                if not isinstance(video_path, str) or not video_path:
                    raise ValueError(f"{self.metadata_path}:{line_number} has no video_path")
                if (
                    not isinstance(begin, int)
                    or not isinstance(end, int)
                    or begin < 0
                    or end <= begin
                ):
                    raise ValueError(f"{self.metadata_path}:{line_number} has invalid bounds")
                self.total_records += 1
                if end - begin < num_frames:
                    self.short_records += 1
                    continue
                self.records.append((video_path, begin, end))
        if not self.records:
            raise ValueError(f"No {num_frames}-frame clips in {self.metadata_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, key) -> torch.Tensor:
        index = int(key.index)
        crop_seed = int(key.crop_seed)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if crop_seed < 0:
            raise ValueError(f"crop_seed must be non-negative, got {crop_seed}")

        video_path, begin, end = self.records[index]
        generator = torch.Generator().manual_seed(crop_seed)
        maximum_offset = end - begin - self.num_frames
        offset = int(torch.randint(maximum_offset + 1, (), generator=generator))
        video = load_consecutive_rgb_frames(
            video_path,
            begin + offset,
            self.num_frames,
        )
        cropped = gnvc_train_preprocess(video, generator=generator)
        return cropped.float().div(127.5).sub(1.0).permute(1, 0, 2, 3)
