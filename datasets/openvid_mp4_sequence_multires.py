from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .openvid_mp4_sequence import TrainingSampleKey
from .transforms import gnvc_train_preprocess


def load_mp4_window(
    video_path: str | Path,
    start_frame: int,
    num_frames: int,
) -> torch.Tensor:
    """Decode consecutive RGB frames as uint8 [T,C,H,W]."""
    video_path = Path(video_path)
    if start_frame < 0:
        raise ValueError(f"start_frame must be non-negative, got {start_frame}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    frames: list[torch.Tensor] = []
    try:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame)):
            raise RuntimeError(
                f"OpenCV could not seek {video_path} to frame {start_frame}"
            )
        actual_position = capture.get(cv2.CAP_PROP_POS_FRAMES)
        if abs(actual_position - start_frame) > 0.5:
            raise RuntimeError(
                f"OpenCV seek mismatch for {video_path}: "
                f"requested={start_frame}, actual={actual_position}"
            )
        frame_size: tuple[int, int] | None = None
        for offset in range(num_frames):
            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise RuntimeError(
                    f"OpenCV stopped at source frame {start_frame + offset}: "
                    f"{video_path}"
                )
            if bgr.ndim != 3 or bgr.shape[2] != 3:
                raise ValueError(f"Expected BGR frame, got {bgr.shape}: {video_path}")
            current_size = (int(bgr.shape[0]), int(bgr.shape[1]))
            if frame_size is None:
                frame_size = current_size
            elif current_size != frame_size:
                raise ValueError(
                    f"Frame size changed from {frame_size} to {current_size}: "
                    f"{video_path}"
                )
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            frames.append(torch.from_numpy(rgb).permute(2, 0, 1))
    finally:
        capture.release()
    return torch.stack(frames, dim=0)


class SeededOpenVidMp4MultiresTrainingDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        dataset_root: str | Path,
        metadata_path: str | Path,
        *,
        num_frames: int = 33,
        spatial_mode: str,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.video_root = self.dataset_root / "videos"
        self.metadata_path = Path(metadata_path)
        self.num_frames = num_frames
        self.spatial_mode = spatial_mode
        if num_frames <= 0 or (num_frames - 1) % 4:
            raise ValueError(
                f"OpenVid sequence length must be a positive 4k+1 value, got {num_frames}"
            )
        if spatial_mode not in {"512", "1080p"}:
            raise ValueError(f"spatial_mode must be 512 or 1080p, got {spatial_mode!r}")
        if not self.video_root.is_dir():
            raise FileNotFoundError(f"Missing OpenVid video root: {self.video_root}")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Missing OpenVid metadata: {self.metadata_path}")

        self.records: list[dict] = []
        self.samples: list[dict] = []
        self.video_paths: list[Path] = []
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"{self.metadata_path}:{line_number} is blank")
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError(
                        f"{self.metadata_path}:{line_number} must contain a JSON object"
                    )
                available = record.get("available_sequence_lengths")
                samples = record.get("samples")
                if (
                    not isinstance(available, list)
                    or not available
                    or any(not isinstance(frames, int) for frames in available)
                ):
                    raise ValueError(
                        f"{self.metadata_path}:{line_number} has invalid "
                        "available_sequence_lengths"
                    )
                if not isinstance(samples, dict) or not samples:
                    raise ValueError(
                        f"{self.metadata_path}:{line_number} has invalid samples"
                    )
                expected_sample_keys = {str(frames) for frames in available}
                if set(samples) != expected_sample_keys:
                    raise ValueError(
                        f"{self.metadata_path}:{line_number} has inconsistent "
                        "available_sequence_lengths and samples"
                    )
                if num_frames not in available:
                    continue
                sample = samples[str(num_frames)]
                if not isinstance(sample, dict):
                    raise TypeError(
                        f"{self.metadata_path}:{line_number} sample {num_frames} "
                        "must be a JSON object"
                    )
                if sample.get("frames") != num_frames:
                    raise ValueError(
                        f"{self.metadata_path}:{line_number} sample {num_frames} "
                        "has the wrong frame count"
                    )
                scene_start = sample.get("scene_start_frame")
                scene_end = sample.get("scene_end_frame_exclusive")
                sample_start = sample.get("sample_start_frame")
                sample_end = sample.get("sample_end_frame_inclusive")
                if not all(
                    isinstance(value, int)
                    for value in (scene_start, scene_end, sample_start, sample_end)
                ):
                    raise ValueError(
                        f"{self.metadata_path}:{line_number} sample {num_frames} "
                        "has non-integer frame bounds"
                    )
                if (
                    scene_start < 0
                    or sample_start < scene_start
                    or sample_end != sample_start + num_frames - 1
                    or sample_end >= scene_end
                ):
                    raise ValueError(
                        f"{self.metadata_path}:{line_number} sample {num_frames} "
                        "has invalid frame bounds"
                    )
                member = record.get("member")
                if not isinstance(member, str) or not member:
                    raise ValueError(
                        f"{self.metadata_path}:{line_number} has no MP4 member"
                    )
                member_path = PurePosixPath(member)
                if (
                    member_path.is_absolute()
                    or ".." in member_path.parts
                    or member_path.suffix.lower() != ".mp4"
                ):
                    raise ValueError(
                        f"{self.metadata_path}:{line_number} has unsafe MP4 member "
                        f"{member!r}"
                    )
                video_path = self.video_root.joinpath(*member_path.parts)
                if not video_path.is_file():
                    raise FileNotFoundError(
                        f"Metadata member {member} has no MP4: {video_path}"
                    )
                self.records.append(record)
                self.samples.append(sample)
                self.video_paths.append(video_path)
        if not self.records:
            raise ValueError(
                f"No OpenVid records support {num_frames} frames in "
                f"{self.metadata_path}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, key: TrainingSampleKey) -> torch.Tensor:
        if not isinstance(key, TrainingSampleKey):
            raise TypeError(f"Expected TrainingSampleKey, got {type(key).__name__}")
        if key.index < 0 or key.index >= len(self):
            raise IndexError(key.index)
        if key.crop_seed < 0:
            raise ValueError(f"crop_seed must be non-negative, got {key.crop_seed}")
        record = self.records[key.index]
        sample = self.samples[key.index]
        video = load_mp4_window(
            self.video_paths[key.index],
            int(sample["sample_start_frame"]),
            self.num_frames,
        )
        expected_size = (int(record["height"]), int(record["width"]))
        if tuple(video.shape[-2:]) != expected_size:
            raise ValueError(
                f"Decoded size {tuple(video.shape[-2:])} does not match metadata "
                f"{expected_size}: {self.video_paths[key.index]}"
            )
        generator = torch.Generator().manual_seed(key.crop_seed)
        if self.spatial_mode == "512":
            video = gnvc_train_preprocess(
                video,
                resize_to=512,
                crop_size=512,
                generator=generator,
            )
        elif tuple(video.shape[-2:]) != (1080, 1920):
            raise ValueError(
                f"1080p phase requires exact 1920x1080 input, got "
                f"{tuple(video.shape[-2:])}: {self.video_paths[key.index]}"
            )
        return video.float().div(127.5).sub(1.0).permute(1, 0, 2, 3)
