from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .openvid_sequence import OpenVidTrainingDataset
from .transforms import gnvc_train_preprocess
from .video_sequence import load_png_sequence


@dataclass(frozen=True)
class TrainingSampleKey:
    index: int
    crop_seed: int


class SeededOpenVidTrainingDataset(OpenVidTrainingDataset):
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
        generator = torch.Generator().manual_seed(key.crop_seed)
        cropped = gnvc_train_preprocess(video, generator=generator)
        return cropped.float().div(127.5).sub(1.0).permute(1, 0, 2, 3)


class ThreadedOpenVidBatchDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        dataset: SeededOpenVidTrainingDataset,
        *,
        threads: int,
    ) -> None:
        if threads <= 0:
            raise ValueError(f"threads must be positive, got {threads}")
        self.dataset = dataset
        self.threads = threads
        self.executor = ThreadPoolExecutor(
            max_workers=threads, thread_name_prefix="openvid-png"
        )
        self.closed = False

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, keys: list[TrainingSampleKey]) -> torch.Tensor:
        if self.closed:
            raise RuntimeError("ThreadedOpenVidBatchDataset is closed")
        if not isinstance(keys, list) or not keys:
            raise TypeError("Expected a non-empty list of TrainingSampleKey values")
        if not all(isinstance(key, TrainingSampleKey) for key in keys):
            raise TypeError("Batch contains a value that is not TrainingSampleKey")
        futures = [self.executor.submit(self.dataset.__getitem__, key) for key in keys]
        return torch.stack([future.result() for future in futures], dim=0)

    def close(self) -> None:
        if self.closed:
            raise RuntimeError("ThreadedOpenVidBatchDataset was closed twice")
        self.executor.shutdown(wait=True, cancel_futures=False)
        self.closed = True
