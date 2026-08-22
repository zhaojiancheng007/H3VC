from __future__ import annotations

import cv2
import torch
from torch.utils.data import DataLoader, Dataset

from training.distributed_runtime import ResumableDistributedBatchSampler


FAST_WORKERS_PER_RANK = 4
FAST_PREFETCH_FACTOR = 2


def _initialize_video_worker(_: int) -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def build_video_dataloader_fast(
    dataset: Dataset[torch.Tensor],
    batch_sampler: ResumableDistributedBatchSampler,
    loader_generator: torch.Generator,
    *,
    num_workers: int,
) -> DataLoader:
    if num_workers != FAST_WORKERS_PER_RANK:
        raise ValueError(
            f"Fast loader requires num_workers={FAST_WORKERS_PER_RANK}, "
            f"got {num_workers}"
        )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        generator=loader_generator,
        prefetch_factor=FAST_PREFETCH_FACTOR,
        persistent_workers=True,
        worker_init_fn=_initialize_video_worker,
    )
