from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .openvid_mp4_sequence import TrainingSampleKey


JSON_MOUNT_ROOT = PurePosixPath("/workspace/gemini")
RESIZE_SHORT_SIDE = 512
CROP_SIZE = 256


def load_mp4_window_resize_crop(
    video_path: str | Path,
    start_frame: int,
    num_frames: int,
    frame_size: tuple[int, int],
    generator: torch.Generator,
) -> torch.Tensor:
    """Decode [T,C,256,256] while discarding each full frame immediately."""
    source_height, source_width = frame_size
    scale = RESIZE_SHORT_SIDE / min(source_height, source_width)
    resized_height = int(math.floor(source_height * scale + 0.5))
    resized_width = int(math.floor(source_width * scale + 0.5))
    top = int(
        torch.randint(resized_height - CROP_SIZE + 1, (), generator=generator)
    )
    left = int(
        torch.randint(resized_width - CROP_SIZE + 1, (), generator=generator)
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    frames: list[np.ndarray] = []
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
        for offset in range(num_frames):
            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise RuntimeError(
                    f"OpenCV stopped at source frame {start_frame + offset}: "
                    f"{video_path}"
                )
            if bgr.ndim != 3 or bgr.shape[2] != 3:
                raise ValueError(f"Expected BGR frame, got {bgr.shape}: {video_path}")
            if tuple(bgr.shape[:2]) != frame_size:
                raise ValueError(
                    f"Decoded size {tuple(bgr.shape[:2])} does not match metadata "
                    f"{frame_size}: {video_path}"
                )
            resized = cv2.resize(
                bgr,
                (resized_width, resized_height),
                interpolation=cv2.INTER_CUBIC,
            )
            cropped = resized[top : top + CROP_SIZE, left : left + CROP_SIZE]
            frames.append(np.ascontiguousarray(cropped[:, :, ::-1]))
    finally:
        capture.release()

    array = np.stack(frames)
    return torch.from_numpy(array).permute(0, 3, 1, 2)


class SeededFastTrainableJsonsVideoDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        dataset_root: str | Path,
        metadata_path: str | Path,
        *,
        num_frames: int = 25,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.metadata_path = Path(metadata_path)
        self.num_frames = num_frames
        if num_frames <= 0:
            raise ValueError(f"Sequence length must be positive, got {num_frames}")
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"Missing dataset root: {self.dataset_root}")
        if not self.metadata_path.is_dir():
            raise FileNotFoundError(f"Missing metadata directory: {self.metadata_path}")

        metadata_files = sorted(
            self.metadata_path.glob("valid_all_chunk_*.json"),
            key=lambda path: int(path.stem.removeprefix("valid_all_chunk_")),
        )
        if not metadata_files:
            raise FileNotFoundError(
                f"No valid_all_chunk_*.json files in {self.metadata_path}"
            )

        self.video_paths: list[Path] = []
        self.video_lengths: list[int] = []
        self.frame_sizes: list[tuple[int, int]] = []
        for metadata_file in metadata_files:
            payload = json.loads(metadata_file.read_text(encoding="utf-8"))
            for record in payload["clips"]:
                video_length = int(record["video_length"])
                if video_length < num_frames:
                    continue
                json_path = PurePosixPath(record["video_path"])
                relative_path = json_path.relative_to(JSON_MOUNT_ROOT)
                if ".." in relative_path.parts or relative_path.suffix.lower() != ".mp4":
                    raise ValueError(f"Unsafe video path in {metadata_file}: {json_path}")
                video_path = self.dataset_root.joinpath(*relative_path.parts)
                self.video_paths.append(video_path)
                self.video_lengths.append(video_length)
                self.frame_sizes.append(
                    (int(record["video_height"]), int(record["video_width"]))
                )
        if not self.video_paths:
            raise ValueError(
                f"No records support {num_frames} frames in {self.metadata_path}"
            )

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, key: TrainingSampleKey) -> torch.Tensor:
        if not isinstance(key, TrainingSampleKey):
            raise TypeError(f"Expected TrainingSampleKey, got {type(key).__name__}")
        if key.index < 0 or key.index >= len(self):
            raise IndexError(key.index)
        if key.crop_seed < 0:
            raise ValueError(f"crop_seed must be non-negative, got {key.crop_seed}")

        generator = torch.Generator().manual_seed(key.crop_seed)
        max_start = self.video_lengths[key.index] - self.num_frames
        start_frame = int(torch.randint(max_start + 1, (), generator=generator))
        video = load_mp4_window_resize_crop(
            self.video_paths[key.index],
            start_frame,
            self.num_frames,
            self.frame_sizes[key.index],
            generator,
        )
        return video.float().div_(127.5).sub_(1.0).permute(1, 0, 2, 3)
