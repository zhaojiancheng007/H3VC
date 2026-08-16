from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from datasets.openvid_mp4_sequence import TrainingSampleKey


DATALOADER_PREFETCH_FACTOR = 2


class ResumableDistributedBatchSampler(BatchSampler):
    def __init__(
        self,
        sampler: DistributedSampler,
        batch_size: int,
        *,
        crop_seed: int,
        dataset_size: int,
    ) -> None:
        super().__init__(sampler, batch_size, drop_last=True)
        if crop_seed < 0 or dataset_size <= 0:
            raise ValueError(
                f"Invalid crop seed/dataset size: {crop_seed}/{dataset_size}"
            )
        self.distributed_sampler = sampler
        self.crop_seed = crop_seed
        self.dataset_size = dataset_size
        self.epoch = 0
        self.start_batch = 0

    @property
    def batches_per_epoch(self) -> int:
        return BatchSampler.__len__(self)

    def set_epoch(self, epoch: int, *, start_batch: int) -> None:
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        if start_batch < 0 or start_batch >= self.batches_per_epoch:
            raise ValueError(
                f"start_batch={start_batch} must be in "
                f"[0,{self.batches_per_epoch - 1}]"
            )
        self.distributed_sampler.set_epoch(epoch)
        self.epoch = epoch
        self.start_batch = start_batch

    def __iter__(self):
        batches = super().__iter__()
        for _ in range(self.start_batch):
            next(batches)
        for batch in batches:
            yield [
                TrainingSampleKey(
                    index=index,
                    crop_seed=(
                        self.crop_seed + self.epoch * self.dataset_size + index
                    )
                    % (2**63 - 1),
                )
                for index in batch
            ]

    def __len__(self) -> int:
        return self.batches_per_epoch - self.start_batch


def build_video_dataloader(
    dataset: Dataset[torch.Tensor],
    batch_sampler: ResumableDistributedBatchSampler,
    loader_generator: torch.Generator,
    *,
    num_workers: int,
) -> DataLoader:
    if num_workers <= 0:
        raise ValueError(f"Video loading requires num_workers > 0, got {num_workers}")
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        generator=loader_generator,
        prefetch_factor=DATALOADER_PREFETCH_FACTOR,
    )


class DistributedEpochIterator:
    def __init__(
        self,
        loader: DataLoader,
        batch_sampler: ResumableDistributedBatchSampler,
        loader_generator: torch.Generator,
        *,
        seed: int,
        rank: int,
        epoch: int,
        batch_in_epoch: int,
    ) -> None:
        self.loader = loader
        self.batch_sampler = batch_sampler
        self.loader_generator = loader_generator
        self.seed = seed
        self.rank = rank
        self.epoch = epoch
        self.batch_in_epoch = batch_in_epoch
        if batch_sampler.batches_per_epoch <= 0:
            raise ValueError("Distributed DataLoader must contain at least one batch")
        if batch_in_epoch < 0 or batch_in_epoch >= batch_sampler.batches_per_epoch:
            raise ValueError(
                f"batch_in_epoch={batch_in_epoch} must be in "
                f"[0,{batch_sampler.batches_per_epoch - 1}]"
            )
        self._reset_iterator(start_batch=batch_in_epoch)

    def _reset_iterator(self, *, start_batch: int) -> None:
        self.batch_sampler.set_epoch(self.epoch, start_batch=start_batch)
        self.loader_generator.manual_seed(
            self.seed + self.rank + self.epoch * 1_000_003
        )
        self.iterator = iter(self.loader)

    def next(self) -> torch.Tensor:
        try:
            batch = next(self.iterator)
        except StopIteration:
            if self.batch_in_epoch != self.batch_sampler.batches_per_epoch:
                raise RuntimeError(
                    f"DataLoader stopped at batch {self.batch_in_epoch}, expected "
                    f"{self.batch_sampler.batches_per_epoch}"
                )
            self.epoch += 1
            self.batch_in_epoch = 0
            self._reset_iterator(start_batch=0)
            batch = next(self.iterator)
        self.batch_in_epoch += 1
        return batch

    def checkpoint_position(self) -> tuple[int, int]:
        if self.batch_in_epoch == self.batch_sampler.batches_per_epoch:
            return self.epoch + 1, 0
        return self.epoch, self.batch_in_epoch


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
    }


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"RNG state is missing fields: {sorted(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device)


def collective_error(local_error: str | None, device: torch.device) -> None:
    failed = torch.tensor(int(local_error is not None), device=device)
    dist.all_reduce(failed, op=dist.ReduceOp.SUM)
    if int(failed.item()) == 0:
        return
    errors: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, local_error)
    details = "; ".join(
        f"rank {rank}: {error}"
        for rank, error in enumerate(errors)
        if error is not None
    )
    raise RuntimeError(f"Distributed invariant failed: {details}")


def all_reduce_mean(values: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values / dist.get_world_size()


def rank_zero_output_setup(
    output_dir: Path,
    *,
    resume: bool,
    resume_step: int,
    rank: int,
) -> None:
    error: str | None = None
    if rank == 0:
        try:
            if not resume and output_dir.exists() and any(output_dir.iterdir()):
                raise FileExistsError(
                    f"Refusing to write into non-empty output directory: {output_dir}"
                )
            output_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = output_dir / "train_metrics.jsonl"
            if resume and metrics_path.is_file():
                lines = [
                    line
                    for line in metrics_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if lines:
                    last_step = int(json.loads(lines[-1])["step"])
                    if last_step != resume_step:
                        raise ValueError(
                            f"Metrics end at step {last_step}, but checkpoint is "
                            f"step {resume_step}"
                        )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    message = [error]
    dist.broadcast_object_list(message, src=0)
    if message[0] is not None:
        raise RuntimeError(f"Output setup failed on rank 0: {message[0]}")
    dist.barrier()


def write_json(path: Path, values: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def append_log(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
