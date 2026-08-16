from __future__ import annotations

import argparse
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
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.reporting import write_json
from tokenizer.stage1_vq import (
    CODEBOOK_SIZE,
    QUANTIZER_GEOMETRY,
    Stage1VQTokenizer,
)
from tokenizer.stage_backbones import (
    TOKENIZER_SPECS,
    build_backbone,
    checkpoint_input_frames_match,
)
from training.config_stage1 import (
    learning_rate_at_step,
    learning_rate_schedule_dict,
    load_stage1_training_config,
    stage1_config_dict,
)
from training.stage_data import (
    DEFAULT_DATASET_FORMAT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_METADATA,
    build_stage_dataset,
)
from training.distributed_runtime import (
    DistributedEpochIterator,
    ResumableDistributedBatchSampler,
    all_reduce_mean,
    append_log,
    build_video_dataloader,
    rank_zero_output_setup,
)
from training.video_losses import chunked_video_lpips, create_lpips_vgg


MSE_WEIGHT = 1.0
LPIPS_WEIGHT = 1.0
LPIPS_FRAME_BATCH_SIZE = 16
CHECKPOINT_INTERVAL = 5000




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one raw-L2 4096-way EMA-VQ tokenizer"
    )
    parser.add_argument("--tokenizer", required=True, choices=tuple(TOKENIZER_SPECS))
    parser.add_argument(
        "--dataset-format",
        default=DEFAULT_DATASET_FORMAT,
        choices=("openvid", "trainable_jsons"),
    )
    parser.add_argument(
        "-c", "--config", default=str(PROJECT_ROOT / "configs/stage1.yaml")
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--uvg-root", default="/luojix/data/UVG")
    parser.add_argument("--wan-repo", default=str(PROJECT_ROOT / "Wan2.1"))
    parser.add_argument(
        "--wan-checkpoint",
        default=str(PROJECT_ROOT / "pretrained_ckpts/Wan2.1/Wan2.1_VAE.pth"),
    )
    parser.add_argument("--cosmos-repo", default=str(PROJECT_ROOT / "Cosmos-Tokenizer"))
    parser.add_argument(
        "--cosmos-encoder",
        default=str(
            PROJECT_ROOT
            / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV4x8x8/encoder.jit"
        ),
    )
    parser.add_argument(
        "--cosmos-decoder",
        default=str(
            PROJECT_ROOT
            / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV4x8x8/decoder.jit"
        ),
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
        default=str(PROJECT_ROOT / "pretrained_ckpts/MiniMax-H3/FL2VA/video_vae"),
    )
    parser.add_argument("--resume")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


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
    evaluation_path: Path,
    model: Stage1VQTokenizer,
    optimizer: torch.optim.Optimizer,
    lpips_metric: torch.nn.Module,
    *,
    rank: int,
    local_rank: int,
    device: torch.device,
) -> DistributedDataParallel:
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    if rank == 0:
        model.to("cpu")
        move_optimizer_state(optimizer, torch.device("cpu"))
        lpips_metric.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        physical_device = (
            str(local_rank)
            if visible_devices is None
            else visible_devices.split(",")[local_rank]
        )
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = physical_device
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            environment.pop(name, None)
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts/evaluate_stage1_tokenizer_uvg.py"),
                "--checkpoint",
                str(checkpoint_path.resolve()),
                "--output",
                str(evaluation_path.resolve()),
                "--uvg-root",
                args.uvg_root,
                "--wan-repo",
                args.wan_repo,
                "--wan-checkpoint",
                args.wan_checkpoint,
                "--cosmos-repo",
                args.cosmos_repo,
                "--cosmos-encoder",
                args.cosmos_encoder,
                "--cosmos-decoder",
                args.cosmos_decoder,
                "--cosmos-cv888-encoder",
                args.cosmos_cv888_encoder,
                "--cosmos-cv888-decoder",
                args.cosmos_cv888_decoder,
                "--minimax-h3-model-dir",
                args.minimax_h3_model_dir,
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
        )

        model.to(device)
        move_optimizer_state(optimizer, device)
        lpips_metric.to(device)
        torch.cuda.empty_cache()

    dist.barrier()
    return DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )


def save_checkpoint(
    path: Path,
    model: Stage1VQTokenizer,
    optimizer: torch.optim.Optimizer,
    tokenizer_name: str,
    input_frames: int,
    latent_channels: int,
    codebook_dim: int,
    config: dict,
    step: int,
) -> None:
    payload = {
        "version": 2,
        "stage": 1,
        "step": step,
        "tokenizer": tokenizer_name,
        "input_frames": input_frames,
        "latent_channels": latent_channels,
        "quantizer_geometry": QUANTIZER_GEOMETRY,
        "codebook_size": CODEBOOK_SIZE,
        "codebook_dim": codebook_dim,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def keep_best_lpips_checkpoint(
    candidate: Path,
    evaluation: Path,
    best_checkpoint: Path,
    best_record: Path,
) -> bool:
    with evaluation.open("r", encoding="utf-8") as handle:
        evaluation_payload = json.load(handle)
    aggregate = evaluation_payload["aggregate"]
    current_lpips = float(aggregate["lpips"])
    best_lpips = math.inf
    if best_record.is_file():
        with best_record.open("r", encoding="utf-8") as handle:
            best_payload = json.load(handle)
        best_lpips = float(best_payload["selection_value"])
    if current_lpips >= best_lpips:
        candidate.unlink()
        return False
    os.replace(candidate, best_checkpoint)
    write_json(
        best_record,
        {
            "schema_version": 1,
            "checkpoint_step": int(evaluation_payload["checkpoint_step"]),
            "selection_metric": "UVG aggregate LPIPS-VGG",
            "selection_value": current_lpips,
            "checkpoint": str(best_checkpoint.resolve()),
            "evaluation": str(evaluation.resolve()),
            "aggregate": aggregate,
        },
    )
    return True


def run(args: argparse.Namespace) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    spec = TOKENIZER_SPECS[args.tokenizer]
    config = load_stage1_training_config(args.config)
    output_dir = Path(args.output_dir)
    config_payload = stage1_config_dict(config)
    resume_payload = None
    start_step = 0
    if args.resume is not None:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        required = {
            "version",
            "stage",
            "step",
            "tokenizer",
            "input_frames",
            "latent_channels",
            "quantizer_geometry",
            "codebook_size",
            "codebook_dim",
            "model",
            "optimizer",
            "config",
        }
        if set(resume_payload) != required:
            raise ValueError(
                f"Stage1 checkpoint fields are {sorted(resume_payload)}, "
                f"expected {sorted(required)}"
            )
        if resume_payload["tokenizer"] != args.tokenizer:
            raise ValueError("Stage1 resume tokenizer changed")
        if not checkpoint_input_frames_match(
            args.tokenizer, resume_payload["input_frames"]
        ):
            raise ValueError(
                f"Stage1 resume input frames changed from "
                f"{resume_payload['input_frames']} to {spec.input_frames}"
            )
        checkpoint_geometry = (
            resume_payload["latent_channels"], resume_payload["codebook_dim"],
            resume_payload["quantizer_geometry"],
        )
        expected_geometry = (
            spec.latent_channels, spec.codebook_dim, QUANTIZER_GEOMETRY,
        )
        if checkpoint_geometry != expected_geometry:
            raise ValueError(f"Stage1 resume geometry changed: {checkpoint_geometry}")
        if resume_payload["config"] != config_payload:
            raise ValueError("Stage1 resume config changed")
        start_step = int(resume_payload["step"])

    random.seed(config.train.seed + rank)
    np.random.seed(config.train.seed + rank)
    torch.manual_seed(config.train.seed + rank)
    torch.cuda.manual_seed(config.train.seed + rank)

    rank_zero_output_setup(
        output_dir,
        resume=resume_payload is not None,
        resume_step=start_step,
        rank=rank,
    )
    dataset = build_stage_dataset(
        args.dataset_format,
        args.dataset_root,
        args.metadata,
        num_frames=spec.input_frames,
        output_dir=output_dir,
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
    loader_generator = torch.Generator()
    loader = build_video_dataloader(
        dataset,
        batch_sampler,
        loader_generator,
        num_workers=config.train.num_workers,
    )
    start_epoch, start_batch = divmod(
        start_step, batch_sampler.batches_per_epoch
    )
    stream = DistributedEpochIterator(
        loader,
        batch_sampler,
        loader_generator,
        seed=config.train.seed,
        rank=rank,
        epoch=start_epoch,
        batch_in_epoch=start_batch,
    )

    backbone = build_backbone(
        args.tokenizer,
        wan_repo=args.wan_repo,
        wan_checkpoint=args.wan_checkpoint,
        cosmos_repo=args.cosmos_repo,
        cosmos_cv488_encoder=args.cosmos_encoder,
        cosmos_cv488_decoder=args.cosmos_decoder,
        cosmos_cv888_encoder=args.cosmos_cv888_encoder,
        cosmos_cv888_decoder=args.cosmos_cv888_decoder,
        minimax_h3_model_dir=args.minimax_h3_model_dir,
    )
    model = Stage1VQTokenizer(
        backbone, latent_channels=spec.latent_channels, codebook_dim=spec.codebook_dim
    ).to(device)
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model"], strict=True)
    ddp_model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate_at_step(config.optimizer, 0),
        betas=config.optimizer.betas,
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
    lpips_metric = create_lpips_vgg(device)

    manifest = {
        "schema_version": 2,
        "stage": 1,
        "tokenizer": args.tokenizer,
        "input_frames": spec.input_frames,
        "latent_channels": spec.latent_channels,
        "codebook": [CODEBOOK_SIZE, spec.codebook_dim],
        "quantizer": QUANTIZER_GEOMETRY,
        "codebook_initialization": "10 Lloyd iterations on the first global latent batch",
        "ema_decay": model.quantizer.decay,
        "loss": {
            "mse_weight": MSE_WEIGHT,
            "lpips_vgg_weight": LPIPS_WEIGHT,
            "lpips_frame_batch_size": LPIPS_FRAME_BATCH_SIZE,
            "lpips_activation_checkpointing": True,
            "commitment_weight": model.quantizer.commitment_weight,
        },
        "checkpoint_and_uvg_interval": config.train.checkpoint_interval,
        "checkpoint_policy": config.train.checkpoint_policy,
        "learning_rate_schedule": learning_rate_schedule_dict(config.optimizer),
        "precision": "BF16 autocast with FP32 VQ assignment and losses",
        "world_size": world_size,
        "batch_size_per_gpu": config.train.batch_size,
        "global_batch_size": config.train.batch_size * world_size,
        "config": config_payload,
    }
    if rank == 0:
        manifest_name = (
            "run_manifest.json"
            if resume_payload is None
            else f"resume_{start_step:07d}.json"
        )
        write_json(output_dir / manifest_name, manifest)

    metrics_path = output_dir / "train_metrics.jsonl"
    log_path = output_dir / "training.log"
    progress = tqdm(
        total=config.train.max_steps,
        initial=start_step,
        desc=f"Stage1 {args.tokenizer} EMA-VQ",
        dynamic_ncols=True,
    ) if rank == 0 else None

    resume_evaluation_path = (
        output_dir / "evaluations" / f"step_{start_step:07d}.json"
    )
    if resume_payload is not None and not resume_evaluation_path.exists():
        del ddp_model
        ddp_model = evaluate_checkpoint_in_subprocess(
            args,
            Path(args.resume),
            resume_evaluation_path,
            model,
            optimizer,
            lpips_metric,
            rank=rank,
            local_rank=local_rank,
            device=device,
        )

    model.train()
    lpips_metric.eval()
    for step in range(start_step + 1, config.train.max_steps + 1):
        synchronize(device)
        started = time.perf_counter()
        video = stream.next().to(device, non_blocking=True)
        learning_rate = learning_rate_at_step(config.optimizer, step)
        optimizer.param_groups[0]["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = ddp_model(video)
            lpips_loss = chunked_video_lpips(
                video,
                output.reconstruction,
                lpips_metric,
                frame_batch_size=LPIPS_FRAME_BATCH_SIZE,
                activation_checkpointing=True,
            )
        mse_loss = F.mse_loss(output.reconstruction.float(), video.float())
        total_loss = (
            MSE_WEIGHT * mse_loss
            + LPIPS_WEIGHT * lpips_loss.float()
            + output.commitment_loss
        )
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters, config.optimizer.grad_clip_norm
        )
        optimizer.step()
        synchronize(device)

        mse = F.mse_loss(output.reconstruction.detach().float(), video.float())
        psnr = 10.0 * torch.log10(4.0 / mse)
        pixels = video.shape[0] * video.shape[2] * video.shape[3] * video.shape[4]
        bpp = output.indices.numel() * 12.0 / pixels
        values = torch.stack(
            (
                total_loss.detach(),
                mse_loss.detach(),
                lpips_loss.detach(),
                output.commitment_loss.detach(),
                output.codebook_loss.detach(),
                psnr,
                torch.tensor(bpp, device=device),
            )
        ).double()
        values = all_reduce_mean(values)
        histogram = torch.bincount(
            output.indices.reshape(-1), minlength=CODEBOOK_SIZE
        ).double()
        dist.all_reduce(histogram)
        probabilities = histogram / histogram.sum()
        active = probabilities > 0
        perplexity = torch.exp(
            -(probabilities[active] * probabilities[active].log()).sum()
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        maximum_latency = torch.tensor(latency_ms, device=device, dtype=torch.float64)
        dist.all_reduce(maximum_latency, op=dist.ReduceOp.MAX)

        metrics = {
            "step": step,
            "psnr": float(values[5]),
            "bpp": float(values[6]),
            "lpips": float(values[2]),
            "latency_ms": float(maximum_latency),
            "learning_rate": learning_rate,
            "loss": {
                "total": float(values[0]),
                "mse": float(values[1]),
                "lpips": float(values[2]),
                "commitment": float(values[3]),
                "codebook_error_diagnostic": float(values[4]),
            },
            "codebook": {
                "active": int(active.sum()),
                "perplexity": float(perplexity),
            },
            "grad_norm": float(grad_norm),
        }
        if rank == 0:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")
            progress.set_postfix(
                loss=f"{metrics['loss']['total']:.4f}",
                psnr=f"{metrics['psnr']:.2f}",
                lpips=f"{metrics['lpips']:.4f}",
                codes=metrics["codebook"]["active"],
            )
            progress.update(1)
            if step == 1 or step % config.train.log_interval == 0:
                append_log(
                    log_path,
                    f"step={step:07d} loss={metrics['loss']['total']:.6f} "
                    f"psnr={metrics['psnr']:.4f} lpips={metrics['lpips']:.6f} "
                    f"active={metrics['codebook']['active']} "
                    f"perplexity={metrics['codebook']['perplexity']:.2f} "
                    f"learning_rate={metrics['learning_rate']:.8g} "
                    f"latency_ms={metrics['latency_ms']:.2f}",
                )

        if step % config.train.checkpoint_interval == 0:
            evaluation_path = output_dir / "evaluations" / f"step_{step:07d}.json"
            checkpoint_path = (
                output_dir / "checkpoints" / f".candidate_step_{step:07d}.pt"
            )
            if rank == 0:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    args.tokenizer,
                    spec.input_frames,
                    spec.latent_channels,
                    spec.codebook_dim,
                    config_payload,
                    step,
                )
            dist.barrier()
            optimizer.zero_grad(set_to_none=True)
            del output, video, lpips_loss, mse_loss, total_loss
            del mse, psnr, values, histogram, probabilities, active, perplexity
            del ddp_model
            ddp_model = evaluate_checkpoint_in_subprocess(
                args,
                checkpoint_path,
                evaluation_path,
                model,
                optimizer,
                lpips_metric,
                rank=rank,
                local_rank=local_rank,
                device=device,
            )
            if rank == 0:
                improved = keep_best_lpips_checkpoint(
                    checkpoint_path,
                    evaluation_path,
                    output_dir / "checkpoints" / "checkpoint_best.pt",
                    output_dir / "best_checkpoint.json",
                )
                append_log(
                    log_path,
                    f"step={step:07d} best_lpips_updated={str(improved).lower()} "
                    f"evaluation={evaluation_path}",
                )
            dist.barrier()
            model.train()
            lpips_metric.eval()

    if rank == 0:
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
