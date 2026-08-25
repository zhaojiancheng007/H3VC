from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import timedelta
import gc
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from entropy_model.model_categorical import CategoricalEntropyModel
from metrics.reporting import write_json
from tokenizer.stage_checkpoint import load_stage1_tokenizer
from training.training_state import DDPTrainingState
from training.config_stage2_cosine import (
    Stage2Config,
    learning_rate_at_step,
    load_stage2_config,
    stage2_config_dict,
)
from training.stage_data_fast import (
    DEFAULT_DATASET_FORMAT,
    build_stage_dataset_fast as build_stage_dataset,
)
from training.checkpoint_config_fast import assert_fast_loader_resume_config
from training.checkpoint_paths import assert_checkpoint_reference
from training.distributed_runtime import (
    DistributedEpochIterator,
    IntervalProgress,
    ResumableDistributedBatchSampler,
    all_reduce_mean,
    append_log,
    capture_rng_state,
    collective_error,
    rank_zero_output_setup,
    restore_rng_state,
)
from training.distributed_runtime_fast import (
    FAST_PREFETCH_FACTOR,
    FAST_WORKERS_PER_RANK,
    build_video_dataloader_fast as build_video_dataloader,
)

from training.evaluation_categorical_distributed import merge_evaluation_shards
from training.periodic_benchmarks import evaluation_environment
from training.periodic_benchmarks_gaussian import (
    distributed_evaluation_command as evaluation_command,
)


CHECKPOINT_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one 4096-way entropy model for a Stage1 VQ tokenizer"
    )
    parser.add_argument(
        "-c",
        "--config",
        default=str(PROJECT_ROOT / "configs/stage2.yaml"),
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--dataset-format",
        default=DEFAULT_DATASET_FORMAT,
        choices=("trainable_jsons",),
    )
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--tokenizer-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--uvg-root", required=True)
    parser.add_argument(
        "--mcl-jcv-root",
        required=True,
    )
    parser.add_argument(
        "--hevc-b-root",
        required=True,
    )
    parser.add_argument("--wan-repo", default=str(PROJECT_ROOT / "Wan2.1"))
    parser.add_argument(
        "--wan-checkpoint",
        default=str(PROJECT_ROOT / "pretrained_ckpts/Wan2.1/Wan2.1_VAE.pth"),
    )
    parser.add_argument("--cosmos-repo", default=str(PROJECT_ROOT / "Cosmos-Tokenizer"))
    parser.add_argument(
        "--cosmos-encoder",
        default=str(PROJECT_ROOT / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV4x8x8/encoder.jit"),
    )
    parser.add_argument(
        "--cosmos-decoder",
        default=str(PROJECT_ROOT / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV4x8x8/decoder.jit"),
    )
    parser.add_argument(
        "--cosmos-cv888-encoder",
        default=str(PROJECT_ROOT / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV8x8x8/encoder.jit"),
    )
    parser.add_argument(
        "--cosmos-cv888-decoder",
        default=str(PROJECT_ROOT / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV8x8x8/decoder.jit"),
    )
    parser.add_argument(
        "--minimax-h3-model-dir",
        required=True,
    )
    parser.add_argument("--resume")
    parser.add_argument(
        "--allow-checkpoint-relocation",
        action="store_true",
        help="Accept explicitly supplied source checkpoints as relocated exact copies",
    )
    parser.add_argument("--stop-after-steps", type=int)
    return parser.parse_args()


def scheduler_scale(config: Stage2Config, scheduler_step: int) -> float:
    schedule_step = min(scheduler_step + 1, config.train.max_steps)
    return (
        learning_rate_at_step(config.optimizer, config.train.max_steps, schedule_step)
        / config.optimizer.learning_rate
    )


def load_checkpoint(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "version",
        "stage",
        "step",
        "tokenizer",
        "tokenizer_checkpoint",
        "config",
        "model",
        "optimizer",
        "scheduler",
        "training_state",
        "rng_states",
    }
    if set(payload) != required:
        raise ValueError(
            f"Stage2 checkpoint fields are {sorted(payload)}, "
            f"expected {sorted(required)}"
        )
    if payload["version"] != CHECKPOINT_VERSION or payload["stage"] != 2:
        raise ValueError("Expected a version-2 Stage2 checkpoint")
    return payload


def move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value):
                state[name] = value.to(device)


def keep_best_bpp_checkpoint(
    candidate: Path,
    evaluation: Path,
    best_checkpoint: Path,
    best_record: Path,
) -> bool:
    with evaluation.open("r", encoding="utf-8") as handle:
        evaluation_payload = json.load(handle)
    current_bpp = float(evaluation_payload["aggregate"]["bpp"])
    best_bpp = math.inf
    if best_record.is_file():
        with best_record.open("r", encoding="utf-8") as handle:
            best_bpp = float(json.load(handle)["selection_value"])
    if current_bpp >= best_bpp:
        candidate.unlink()
        return False
    os.replace(candidate, best_checkpoint)
    write_json(
        best_record,
        {
            "schema_version": 1,
            "checkpoint_step": int(evaluation_payload["checkpoint_step"]),
            "selection_metric": "UVG aggregate estimated BPP",
            "selection_value": current_bpp,
            "checkpoint": str(best_checkpoint.resolve()),
            "evaluation": str(evaluation.resolve()),
            "aggregate": evaluation_payload["aggregate"],
        },
    )
    return True


def evaluate_checkpoint_in_subprocess(
    args: argparse.Namespace,
    checkpoint_path: Path,
    evaluation_dir: Path,
    tokenizer: torch.nn.Module,
    model: CategoricalEntropyModel,
    optimizer: torch.optim.Optimizer,
    *,
    rank: int,
    local_rank: int,
    world_size: int,
    device: torch.device,
) -> DistributedDataParallel:
    cpu = torch.device("cpu")
    tokenizer.to(cpu)
    model.to(cpu)
    move_optimizer_state(optimizer, cpu)
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    subprocess.run(
        evaluation_command(
            args, 2, checkpoint_path, evaluation_dir, rank, world_size, PROJECT_ROOT
        ),
        cwd=PROJECT_ROOT,
        env=evaluation_environment(local_rank),
        check=True,
    )
    dist.barrier()
    if rank == 0:
        shards = []
        for shard_rank in range(world_size):
            shard_path = evaluation_dir / "shards" / f"rank_{shard_rank:02d}.json"
            with shard_path.open("r", encoding="utf-8") as handle:
                shards.append(json.load(handle))
        evaluations = merge_evaluation_shards(shards)
        for dataset, evaluation in evaluations.items():
            write_json(evaluation_dir / f"{dataset}.json", evaluation)
    dist.barrier()

    tokenizer.to(device)
    model.to(device)
    move_optimizer_state(optimizer, device)
    torch.cuda.empty_cache()
    return DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )


def save_checkpoint(
    path: Path,
    model: CategoricalEntropyModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: Stage2Config,
    tokenizer_name: str,
    tokenizer_checkpoint: Path,
    *,
    step: int,
    stream: DistributedEpochIterator,
    world_size: int,
    global_batch_size: int,
    rank: int,
    device: torch.device,
) -> None:
    rng_states = [None] * world_size
    dist.all_gather_object(rng_states, capture_rng_state(device))
    positions = [None] * world_size
    dist.all_gather_object(positions, stream.checkpoint_position())
    if len(set(positions)) != 1:
        raise RuntimeError(f"Ranks disagree on DataLoader position: {positions}")
    if rank == 0:
        state = DDPTrainingState(
            step=step,
            epoch=positions[0][0],
            batch_in_epoch=positions[0][1],
            max_steps=config.train.max_steps,
            world_size=world_size,
            global_batch_size=global_batch_size,
        )
        payload = {
            "version": CHECKPOINT_VERSION,
            "stage": 2,
            "step": step,
            "tokenizer": tokenizer_name,
            "tokenizer_checkpoint": str(tokenizer_checkpoint),
            "config": stage2_config_dict(config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "training_state": asdict(state),
            "rng_states": rng_states,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
    dist.barrier()


def run(args: argparse.Namespace) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    config = load_stage2_config(args.config)
    if config.train.batch_size != 2:
        raise ValueError(
            f"Canonical Stage2 requires batch_size=2 per GPU, got {config.train.batch_size}"
        )
    config_payload = stage2_config_dict(config)
    global_batch_size = config.train.batch_size * world_size

    random.seed(config.train.seed + rank)
    np.random.seed(config.train.seed + rank)
    torch.manual_seed(config.train.seed)
    torch.cuda.manual_seed(config.train.seed)

    loaded = load_stage1_tokenizer(
        args.tokenizer_checkpoint,
        wan_repo=args.wan_repo,
        wan_checkpoint=args.wan_checkpoint,
        cosmos_repo=args.cosmos_repo,
        cosmos_encoder=args.cosmos_encoder,
        cosmos_decoder=args.cosmos_decoder,
        cosmos_cv888_encoder=args.cosmos_cv888_encoder,
        cosmos_cv888_decoder=args.cosmos_cv888_decoder,
        minimax_h3_model_dir=args.minimax_h3_model_dir,
    )
    if config.model.codebook_dim != loaded.codebook_dim:
        raise ValueError(
            f"Entropy config codebook_dim={config.model.codebook_dim}, "
            f"Stage1 requires {loaded.codebook_dim}"
        )
    dataset = build_stage_dataset(
        args.dataset_format, args.dataset_root, args.metadata,
        num_frames=loaded.input_frames, output_dir=args.output_dir,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.train.seed,
        drop_last=True,
    )
    batch_sampler = ResumableDistributedBatchSampler(
        sampler,
        config.train.batch_size,
        crop_seed=config.train.seed,
        dataset_size=len(dataset),
    )
    generator = torch.Generator()
    loader = build_video_dataloader(
        dataset, batch_sampler, generator, num_workers=config.train.num_workers
    )

    tokenizer = loaded.model.to(device).eval().requires_grad_(False)
    model = CategoricalEntropyModel(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        betas=config.optimizer.betas,
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: scheduler_scale(config, step),
    )

    resume = None if args.resume is None else load_checkpoint(args.resume)
    checkpoint_relocated = False
    start_step = 0
    start_epoch = 0
    start_batch = 0
    if resume is not None:
        assert_fast_loader_resume_config(
            resume["config"], config_payload, stage="Stage2"
        )
        if resume["tokenizer"] != loaded.name:
            raise ValueError("Stage2 resume tokenizer changed")
        checkpoint_relocated = assert_checkpoint_reference(
            resume["tokenizer_checkpoint"],
            loaded.checkpoint,
            label="Stage2 resume Stage1 checkpoint",
            allow_relocation=args.allow_checkpoint_relocation,
        )
        state = DDPTrainingState(**resume["training_state"])
        if state.world_size != world_size or state.global_batch_size != global_batch_size:
            raise ValueError("Stage2 resume distributed batch changed")
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        start_step = state.step
        start_epoch = state.epoch
        start_batch = state.batch_in_epoch

    ddp_model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    output_dir = Path(args.output_dir)
    rank_zero_output_setup(
        output_dir, resume=resume is not None, resume_step=start_step, rank=rank
    )
    if resume is not None:
        restore_rng_state(resume["rng_states"][rank], device)
    stream = DistributedEpochIterator(
        loader,
        batch_sampler,
        generator,
        seed=config.train.seed,
        rank=rank,
        epoch=start_epoch,
        batch_in_epoch=start_batch,
    )
    stop_step = config.train.max_steps
    if args.stop_after_steps is not None:
        stop_step = min(stop_step, start_step + args.stop_after_steps)

    if rank == 0:
        write_json(
            output_dir / ("run_manifest.json" if resume is None else f"resume_{start_step:07d}.json"),
            {
                "schema_version": 2,
                "stage": 2,
                "tokenizer": loaded.name,
                "input_frames": loaded.input_frames,
                "latent_channels": loaded.latent_channels,
                "codebook_dim": loaded.codebook_dim,
                "tokenizer_checkpoint": str(loaded.checkpoint),
                "tokenizer_step": loaded.step,
                "tokenizer_frozen": True,
                "prediction_target": "one 4096-way index per latent position",
                "world_size": world_size,
                "batch_size_per_gpu": config.train.batch_size,
                "global_batch_size": global_batch_size,
                "checkpoint_policy": config.train.checkpoint_policy,
                "learning_rate_scheduler": "cosine_annealing",
                "checkpoint_relocation_allowed": args.allow_checkpoint_relocation,
                "checkpoint_relocated": checkpoint_relocated,
                "data_loader": {
                    "implementation": "opencv_worker_resize_crop",
                    "workers_per_rank": FAST_WORKERS_PER_RANK,
                    "prefetch_factor": FAST_PREFETCH_FACTOR,
                    "persistent_workers": True,
                },
                "config": config_payload,
            },
        )
    metrics_path = output_dir / "train_metrics.jsonl"
    log_path = output_dir / "training.log"
    progress = IntervalProgress(
        total=stop_step,
        initial=start_step,
        desc=f"Stage2 {loaded.name} entropy",
        interval=config.train.log_interval,
    ) if rank == 0 else None
    model.train()

    for step in range(start_step + 1, stop_step + 1):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        video = stream.next().to(device, non_blocking=True)
        learning_rate = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=config.train.bf16
        ):
            latent = tokenizer.encode(video)
            projected = tokenizer.pre_quant(latent)
            quantized, indices, _, _ = tokenizer.quantizer(projected)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.train.bf16):
            output = ddp_model(quantized, indices)
        output.bits_per_index.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.optimizer.grad_clip_norm
        )
        collective_error(
            None if torch.isfinite(grad_norm) else f"non-finite gradient at step {step}",
            device,
        )
        optimizer.step()
        scheduler.step()
        pixels = video.shape[0] * video.shape[2] * video.shape[3] * video.shape[4]
        bpp = output.estimated_bits.sum().detach().float() / pixels
        reduced = all_reduce_mean(
            torch.stack((output.bits_per_index.detach().float(), bpp, grad_norm.float()))
        )
        torch.cuda.synchronize(device)
        latency_ms = torch.tensor(
            (time.perf_counter() - started) * 1000.0, device=device
        )
        dist.all_reduce(latency_ms, op=dist.ReduceOp.MAX)
        metrics = {
            "step": step,
            "bpp": float(reduced[1]),
            "bits_per_index": float(reduced[0]),
            "latency_ms": float(latency_ms),
            "grad_norm": float(reduced[2]),
            "learning_rate": learning_rate,
        }
        if rank == 0:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")
            progress.update(1)
            progress.set_postfix(
                bpp=f"{metrics['bpp']:.4f}",
                bits=f"{metrics['bits_per_index']:.3f}",
            )
            if step == 1 or step % config.train.log_interval == 0:
                append_log(
                    log_path,
                    f"step={step:07d} bpp={metrics['bpp']:.6f} "
                    f"bits_per_index={metrics['bits_per_index']:.6f} "
                    f"latency_ms={metrics['latency_ms']:.3f}",
                )

        if step % config.train.checkpoint_interval == 0:
            evaluation_dir = output_dir / "evaluations" / f"step_{step:07d}"
            uvg_evaluation_path = evaluation_dir / "uvg.json"
            checkpoint_path = (
                output_dir / "checkpoints" / f".candidate_step_{step:07d}.pt"
            )
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                config,
                loaded.name,
                loaded.checkpoint,
                step=step,
                stream=stream,
                world_size=world_size,
                global_batch_size=global_batch_size,
                rank=rank,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            del output, video, latent, projected, quantized, indices
            del bpp, reduced, latency_ms, grad_norm
            del ddp_model
            ddp_model = evaluate_checkpoint_in_subprocess(
                args,
                checkpoint_path,
                evaluation_dir,
                tokenizer,
                model,
                optimizer,
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                device=device,
            )
            if rank == 0:
                improved = keep_best_bpp_checkpoint(
                    checkpoint_path,
                    uvg_evaluation_path,
                    output_dir / "checkpoints" / "checkpoint_best.pt",
                    output_dir / "best_checkpoint.json",
                )
                append_log(
                    log_path,
                    f"step={step:07d} best_bpp_updated={str(improved).lower()} "
                    f"evaluation={evaluation_dir}",
                )
            dist.barrier()
            tokenizer.eval()
            model.train()

    if progress is not None:
        progress.close()


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl", timeout=timedelta(hours=6))
    try:
        run(args)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
