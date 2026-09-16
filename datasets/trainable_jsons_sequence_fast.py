from __future__ import annotations

import json
import math
import os
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from .openvid_mp4_sequence import TrainingSampleKey


RESIZE_SHORT_SIDE = 512
CROP_SIZE = 256
MAX_DECODE_ATTEMPTS = 16


class VideoDecodeError(RuntimeError):
    pass


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
        raise VideoDecodeError(f"OpenCV could not open {video_path}")
    frames: list[np.ndarray] = []
    try:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame)):
            raise VideoDecodeError(
                f"OpenCV could not seek {video_path} to frame {start_frame}"
            )
        actual_position = capture.get(cv2.CAP_PROP_POS_FRAMES)
        if abs(actual_position - start_frame) > 0.5:
            raise VideoDecodeError(
                f"OpenCV seek mismatch for {video_path}: "
                f"requested={start_frame}, actual={actual_position}"
            )
        for offset in range(num_frames):
            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise VideoDecodeError(
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
        rejection_log: str | Path | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.metadata_path = Path(metadata_path)
        self.num_frames = num_frames
        self.rejection_log = None if rejection_log is None else Path(rejection_log)
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
                relative_path = json_path
                if relative_path.is_absolute() or ".." in relative_path.parts or relative_path.suffix.lower() != ".mp4":
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
        candidate_index = key.index
        rejected_indices: set[int] = set()
        for attempt in range(1, MAX_DECODE_ATTEMPTS + 1):
            max_start = self.video_lengths[candidate_index] - self.num_frames
            start_frame = int(torch.randint(max_start + 1, (), generator=generator))
            try:
                video = load_mp4_window_resize_crop(
                    self.video_paths[candidate_index],
                    start_frame,
                    self.num_frames,
                    self.frame_sizes[candidate_index],
                    generator,
                )
            except VideoDecodeError as error:
                if self.rejection_log is None:
                    raise
                rejected_indices.add(candidate_index)
                replacement_index = None
                if attempt < MAX_DECODE_ATTEMPTS and len(rejected_indices) < len(self):
                    replacement_rank = int(
                        torch.randint(
                            len(self) - len(rejected_indices),
                            (),
                            generator=generator,
                        )
                    )
                    replacement_index = replacement_rank
                    for rejected_index in sorted(rejected_indices):
                        if replacement_index >= rejected_index:
                            replacement_index += 1

                worker = get_worker_info()
                record = {
                    "event": "rejected_training_sample",
                    "rank": int(os.environ["RANK"]),
                    "worker": worker.id if worker is not None else 0,
                    "requested_index": key.index,
                    "rejected_index": candidate_index,
                    "replacement_index": replacement_index,
                    "attempt": attempt,
                    "crop_seed": key.crop_seed,
                    "start_frame": start_frame,
                    "video_path": str(self.video_paths[candidate_index]),
                    "error": str(error),
                }
                line = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode()
                descriptor = os.open(
                    self.rejection_log,
                    os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                    0o644,
                )
                try:
                    os.write(descriptor, line)
                finally:
                    os.close(descriptor)

                if replacement_index is None:
                    raise RuntimeError(
                        f"Failed to decode {attempt} deterministic candidates for "
                        f"requested dataset index {key.index}"
                    ) from error
                candidate_index = replacement_index
                continue

            return video.float().div_(127.5).sub_(1.0).permute(1, 0, 2, 3)

        raise AssertionError("unreachable")
