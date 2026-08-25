from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from entropy_model.model_categorical import CategoricalEntropyModel
from metrics.reporting import write_json
from tokenizer.stage3_vq import JointVQModel
from tokenizer.stage_checkpoint import load_stage1_tokenizer
from training.training_state import DDPTrainingState
from training.config_stage3_cosine import (
    Stage3Config,
    cosine_learning_rates_at_step,
    load_stage3_config,
    stage3_config_dict,
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
from training.evaluation_categorical_distributed import (
    merge_evaluation_shards,
    replace_reconstruction_archive,
)
from training.periodic_benchmarks import evaluation_environment
from training.periodic_benchmarks_gaussian import (
    distributed_evaluation_command as evaluation_command,
)
from training.video_losses import chunked_video_lpips, create_lpips_vgg


CHECKPOINT_VERSION = 3
LPIPS_FRAME_BATCH_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jointly train a Stage1 VQ tokenizer and Stage2 entropy model"
    )
    parser.add_argument(
        "-c",
        "--config",
        default=str(PROJECT_ROOT / "configs/stage3.yaml"),
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument(
        "--dataset-format",
        default=DEFAULT_DATASET_FORMAT,
        choices=("trainable_jsons",),
    )
    parser.add_argument("--tokenizer-checkpoint", required=True)
    parser.add_argument("--entropy-checkpoint", required=True)
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
    parser.add_argument("--resume")
    parser.add_argument("--init-checkpoint")
    parser.add_argument(
        "--allow-checkpoint-relocation",
        action="store_true",
        help="Accept explicitly supplied source checkpoints as relocated exact copies",
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
    parser.add_argument("--stop-after-steps", type=int)
    return parser.parse_args()


def rate_parameters(step: int, config: Stage3Config) -> tuple[float, float]:
    return config.loss.rate_weight, config.loss.temperature


def estimated_bpp(estimated_bits: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
    pixels = video.shape[0] * video.shape[2] * video.shape[3] * video.shape[4]
    return estimated_bits.sum() / pixels


def load_stage2_checkpoint(path: str | Path) -> dict:
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
    if payload["version"] != 2 or payload["stage"] != 2:
        raise ValueError("Expected a version-2 Stage2 checkpoint")
    return payload


def load_stage3_checkpoint(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    common = {
        "version",
        "stage",
        "step",
        "tokenizer",
        "tokenizer_checkpoint",
        "entropy_checkpoint",
        "config",
        "model",
        "optimizer",
        "training_state",
        "rng_states",
    }
    version = int(payload.get("version", -1))
    required = (
        common | {"discriminator", "discriminator_optimizer"}
        if version == 2
        else common
    )
    if set(payload) != required:
        raise ValueError(
            f"Stage3 checkpoint fields are {sorted(payload)}, "
            f"expected {sorted(required)}"
        )
    if version not in (2, CHECKPOINT_VERSION) or payload["stage"] != 3:
        raise ValueError("Expected a version-2 or version-3 Stage3 checkpoint")
    return payload


def move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value):
                state[name] = value.to(device)


def evaluate_checkpoint_in_subprocess(
    args: argparse.Namespace,
    checkpoint_path: Path,
    evaluation_dir: Path,
    model: JointVQModel,
    optimizer: torch.optim.Optimizer,
    lpips_metric: torch.nn.Module,
    *,
    rank: int,
    local_rank: int,
    world_size: int,
    device: torch.device,
) -> DistributedDataParallel:
    cpu = torch.device("cpu")
    model.to(cpu)
    move_optimizer_state(optimizer, cpu)
    lpips_metric.to(cpu)
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    subprocess.run(
        evaluation_command(
            args, 3, checkpoint_path, evaluation_dir, rank, world_size, PROJECT_ROOT
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
            evaluation["protocol"]["reconstruction"] = f"../recon/{dataset}"
            write_json(evaluation_dir / f"{dataset}.json", evaluation)
        replace_reconstruction_archive(
            evaluation_dir / "recon", evaluation_dir.parent / "recon"
        )
    dist.barrier()

    model.to(device)
    move_optimizer_state(optimizer, device)
    lpips_metric.to(device)
    torch.cuda.empty_cache()
    ddp_model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    return ddp_model


def save_checkpoint(
    path: Path,
    model: JointVQModel,
    optimizer: torch.optim.Optimizer,
    config: Stage3Config,
    tokenizer_name: str,
    tokenizer_checkpoint: Path,
    entropy_checkpoint: Path,
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
            "stage": 3,
            "step": step,
            "tokenizer": tokenizer_name,
            "tokenizer_checkpoint": str(tokenizer_checkpoint),
            "entropy_checkpoint": str(entropy_checkpoint),
            "config": stage3_config_dict(config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "training_state": asdict(state),
            "rng_states": rng_states,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
    dist.barrier()


def scheduler_scale(config: Stage3Config, scheduler_step: int, group: int) -> float:
    schedule_step = min(scheduler_step + 1, config.train.max_steps)
    learning_rates = cosine_learning_rates_at_step(config, schedule_step)
    initial_rates = (
        config.optimizer.tokenizer_learning_rate,
        config.optimizer.entropy_learning_rate,
    )
    return learning_rates[group] / initial_rates[group]


def run(args: argparse.Namespace) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    config = load_stage3_config(args.config)
    config_payload = stage3_config_dict(config)
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
            f"Stage3 codebook_dim={config.model.codebook_dim}, "
            f"Stage1 requires {loaded.codebook_dim}"
        )
    global_batch_size = config.train.batch_size * world_size

    random.seed(config.train.seed + rank)
    np.random.seed(config.train.seed + rank)
    torch.manual_seed(config.train.seed)
    torch.cuda.manual_seed(config.train.seed)

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

    entropy_checkpoint = Path(args.entropy_checkpoint).resolve(strict=True)
    stage2 = load_stage2_checkpoint(entropy_checkpoint)
    if stage2["tokenizer"] != loaded.name:
        raise ValueError("Stage1 and Stage2 tokenizer types differ")
    stage2_source_relocated = assert_checkpoint_reference(
        stage2["tokenizer_checkpoint"],
        loaded.checkpoint,
        label="Stage2 source Stage1 checkpoint",
        allow_relocation=args.allow_checkpoint_relocation,
    )
    if stage2["config"]["model"] != asdict(config.model):
        raise ValueError("Stage2 and Stage3 entropy model configs differ")

    entropy_model = CategoricalEntropyModel(config.model)
    entropy_model.load_state_dict(stage2["model"], strict=True)
    model = JointVQModel(loaded.model, entropy_model)
    initialization_relocated = False
    initialization = (
        None if args.init_checkpoint is None
        else load_stage3_checkpoint(args.init_checkpoint)
    )
    if initialization is not None:
        if initialization["tokenizer"] != loaded.name:
            raise ValueError("Initialization and Stage1 tokenizer types differ")
        initialization_relocated = assert_checkpoint_reference(
            initialization["tokenizer_checkpoint"],
            loaded.checkpoint,
            label="Stage3 initialization Stage1 checkpoint",
            allow_relocation=args.allow_checkpoint_relocation,
        )
        initialization_relocated |= assert_checkpoint_reference(
            initialization["entropy_checkpoint"],
            entropy_checkpoint,
            label="Stage3 initialization Stage2 checkpoint",
            allow_relocation=args.allow_checkpoint_relocation,
        )
        if initialization["config"]["model"] != asdict(config.model):
            raise ValueError("Initialization and cosine Stage3 model configs differ")
        model.load_state_dict(initialization["model"], strict=True)
    resume = None if args.resume is None else load_stage3_checkpoint(args.resume)
    resume_relocated = False
    start_step = 0
    start_epoch = 0
    start_batch = 0
    if resume is not None:
        assert_fast_loader_resume_config(
            resume["config"], config_payload, stage="Stage3"
        )
        if resume["tokenizer"] != loaded.name:
            raise ValueError("Stage3 resume tokenizer changed")
        resume_relocated = assert_checkpoint_reference(
            resume["tokenizer_checkpoint"],
            loaded.checkpoint,
            label="Stage3 resume Stage1 checkpoint",
            allow_relocation=args.allow_checkpoint_relocation,
        )
        resume_relocated |= assert_checkpoint_reference(
            resume["entropy_checkpoint"],
            entropy_checkpoint,
            label="Stage3 resume Stage2 checkpoint",
            allow_relocation=args.allow_checkpoint_relocation,
        )
        state = DDPTrainingState(**resume["training_state"])
        if state.world_size != world_size or state.global_batch_size != global_batch_size:
            raise ValueError("Stage3 resume distributed batch changed")
        model.load_state_dict(resume["model"], strict=True)
        start_step = state.step
        start_epoch = state.epoch
        start_batch = state.batch_in_epoch

    model = model.to(device)
    tokenizer_parameters = [
        parameter for parameter in model.tokenizer.parameters() if parameter.requires_grad
    ]
    entropy_parameters = list(model.entropy_model.parameters())
    optimizer = torch.optim.Adam(
        [
            {
                "params": tokenizer_parameters,
                "lr": config.optimizer.tokenizer_learning_rate,
            },
            {
                "params": entropy_parameters,
                "lr": config.optimizer.entropy_learning_rate,
            },
        ],
        betas=config.optimizer.betas,
    )
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            lambda scheduler_step: scheduler_scale(config, scheduler_step, 0),
            lambda scheduler_step: scheduler_scale(config, scheduler_step, 1),
        ],
        last_epoch=start_step - 1,
    )

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
                "stage": 3,
                "tokenizer": loaded.name,
                "input_frames": loaded.input_frames,
                "latent_channels": loaded.latent_channels,
                "codebook_dim": loaded.codebook_dim,
                "tokenizer_checkpoint": str(loaded.checkpoint),
                "tokenizer_step": loaded.step,
                "entropy_checkpoint": str(entropy_checkpoint),
                "entropy_step": int(stage2["step"]),
                "codebook": [4096, loaded.codebook_dim],
                "codebook_frozen": True,
                "world_size": world_size,
                "batch_size_per_gpu": config.train.batch_size,
                "global_batch_size": global_batch_size,
                "precision": "BF16 autocast with FP32 parameters, VQ assignment, and losses",
                "learning_rate_scheduler": "cosine_annealing",
                "rate_weight": config.loss.rate_weight,
                "temperature": config.loss.temperature,
                "discriminator_training": "disabled",
                "checkpoint_relocation_allowed": args.allow_checkpoint_relocation,
                "checkpoint_relocated": (
                    stage2_source_relocated
                    or initialization_relocated
                    or resume_relocated
                ),
                "data_loader": {
                    "implementation": "opencv_worker_resize_crop",
                    "workers_per_rank": FAST_WORKERS_PER_RANK,
                    "prefetch_factor": FAST_PREFETCH_FACTOR,
                    "persistent_workers": True,
                },
                "initialization_checkpoint": (
                    None if args.init_checkpoint is None
                    else str(Path(args.init_checkpoint).resolve())
                ),
                "config": config_payload,
            },
        )
    metrics_path = output_dir / "train_metrics.jsonl"
    log_path = output_dir / "training.log"
    progress = IntervalProgress(
        total=stop_step,
        initial=start_step,
        desc=f"Stage3 {loaded.name} joint",
        interval=config.train.log_interval,
    ) if rank == 0 else None
    lpips_metric = create_lpips_vgg(device)
    trainable_parameters = [*tokenizer_parameters, *entropy_parameters]
    model.train()

    for step in range(start_step + 1, stop_step + 1):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        video = stream.next().to(device, non_blocking=True)
        rate_weight, temperature = rate_parameters(step, config)
        tokenizer_learning_rate = optimizer.param_groups[0]["lr"]
        entropy_learning_rate = optimizer.param_groups[1]["lr"]
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.train.bf16):
            output = ddp_model(video, temperature=temperature)
            lpips_loss = chunked_video_lpips(
                video,
                output.tokenizer.reconstruction,
                lpips_metric,
                frame_batch_size=LPIPS_FRAME_BATCH_SIZE,
                activation_checkpointing=True,
            )
        mse_loss = F.mse_loss(
            output.tokenizer.reconstruction.float(), video.float()
        )
        commitment_loss = F.mse_loss(
            output.tokenizer.projected,
            output.tokenizer.selected.detach(),
        )
        soft_bpp = estimated_bpp(output.entropy.soft_estimated_bits, video)
        hard_bpp = estimated_bpp(output.entropy.hard_estimated_bits, video)
        total_loss = (
            config.loss.mse_weight * mse_loss
            + config.loss.lpips_weight * lpips_loss.float()
            + config.loss.commitment_weight * commitment_loss
            + rate_weight * soft_bpp
        )
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters, config.optimizer.grad_clip_norm
        )
        collective_error(
            None if torch.isfinite(grad_norm) else f"non-finite gradient at step {step}",
            device,
        )
        optimizer.step()
        scheduler.step()

        psnr = 10.0 * torch.log10(4.0 / mse_loss.detach())
        histogram = torch.bincount(
            output.tokenizer.indices.reshape(-1), minlength=4096
        ).double()
        dist.all_reduce(histogram)
        probabilities = histogram / histogram.sum()
        active = probabilities > 0
        perplexity = torch.exp(
            -(probabilities[active] * probabilities[active].log()).sum()
        )
        scalars = all_reduce_mean(
            torch.stack(
                (
                    total_loss.detach(),
                    mse_loss.detach(),
                    lpips_loss.detach().float(),
                    commitment_loss.detach(),
                    soft_bpp.detach(),
                    hard_bpp.detach(),
                    psnr,
                    grad_norm.detach().float(),
                )
            )
        )
        torch.cuda.synchronize(device)
        latency_ms = torch.tensor(
            (time.perf_counter() - started) * 1000.0, device=device
        )
        dist.all_reduce(latency_ms, op=dist.ReduceOp.MAX)
        metrics = {
            "step": step,
            "psnr": float(scalars[6]),
            "bpp": float(scalars[5]),
            "soft_bpp": float(scalars[4]),
            "lpips": float(scalars[2]),
            "latency_ms": float(latency_ms),
            "loss": {
                "total": float(scalars[0]),
                "mse": float(scalars[1]),
                "lpips": float(scalars[2]),
                "commitment": float(scalars[3]),
            },
            "rate_weight": rate_weight,
            "temperature": temperature,
            "tokenizer_learning_rate": tokenizer_learning_rate,
            "entropy_learning_rate": entropy_learning_rate,
            "discriminator_training": False,
            "codebook": {
                "active": int(active.sum()),
                "perplexity": float(perplexity),
            },
            "grad_norm": float(scalars[7]),
        }
        if rank == 0:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")
            progress.update(1)
            progress.set_postfix(
                bpp=f"{metrics['bpp']:.4f}",
                lpips=f"{metrics['lpips']:.3f}",
                psnr=f"{metrics['psnr']:.2f}",
            )
            if step == 1 or step % config.train.log_interval == 0:
                append_log(
                    log_path,
                    f"step={step:07d} psnr={metrics['psnr']:.4f} "
                    f"bpp={metrics['bpp']:.6f} lpips={metrics['lpips']:.6f} "
                    f"lambda={rate_weight:.4f} tau={temperature:.4f} "
                    f"tokenizer_lr={tokenizer_learning_rate:.8g} "
                    f"entropy_lr={entropy_learning_rate:.8g} "
                    f"latency_ms={metrics['latency_ms']:.3f}",
                )

        if step % config.train.checkpoint_interval == 0:
            checkpoint_path = output_dir / "checkpoints" / f"step_{step:07d}.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                config,
                loaded.name,
                loaded.checkpoint,
                entropy_checkpoint,
                step=step,
                stream=stream,
                world_size=world_size,
                global_batch_size=global_batch_size,
                rank=rank,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            del output, video, lpips_loss, mse_loss, commitment_loss
            del soft_bpp, hard_bpp, total_loss, scalars, latency_ms
            del histogram, probabilities, active, perplexity, grad_norm, psnr
            del ddp_model
            ddp_model = evaluate_checkpoint_in_subprocess(
                args,
                checkpoint_path,
                output_dir / "evaluations" / f"step_{step:07d}",
                model,
                optimizer,
                lpips_metric,
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                device=device,
            )
            model.train()
            lpips_metric.eval()

    if progress is not None:
        progress.close()


def main() -> None:
    args = parse_args()
    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError("Use either --resume or --init-checkpoint, not both")
    dist.init_process_group("nccl", timeout=timedelta(hours=6))
    try:
        run(args)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
