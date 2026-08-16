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
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.video_sequence import load_png_sequence
from metrics.lpips import sequence_lpips
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
from training.config_stage1 import load_stage1_training_config
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
from training.uvg_protocol import (
    finish_uvg_reconstruction,
    prepare_uvg_input,
    uvg_metric_frames,
    uvg_source_frames,
)
from training.video_losses import chunked_video_lpips, create_lpips_vgg


MSE_WEIGHT = 1.0
LPIPS_WEIGHT = 1.0
LPIPS_FRAME_BATCH_SIZE = 16
CHECKPOINT_INTERVAL = 5000
UVG_FRAMES = 96
UVG_SEQUENCES = (
    "Beauty",
    "Bosphorus",
    "HoneyBee",
    "Jockey",
    "ReadySteadyGo",
    "ShakeNDry",
    "YachtRide",
)


def learning_rate_at_step(
    base_learning_rate: float,
    step: int,
    *,
    interval_steps: int,
    decay_factor: float,
    stop_step: int,
) -> float:
    completed_intervals = min(step, stop_step) // interval_steps
    return base_learning_rate * decay_factor**completed_intervals


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
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/stage1_tokenizer.yaml"))
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


@torch.inference_mode()
def sequence_dists(
    reference: torch.Tensor,
    reconstruction: torch.Tensor,
    metric: torch.nn.Module,
    device: torch.device,
) -> float:
    total = 0.0
    for frame in range(reference.shape[0]):
        value = metric(
            reference[frame : frame + 1].to(device),
            reconstruction[frame : frame + 1].to(device),
            require_grad=False,
            batch_average=False,
        )
        total += float(value.item())
    return total / reference.shape[0]


@torch.inference_mode()
def evaluate_uvg(
    model: Stage1VQTokenizer,
    tokenizer_name: str,
    uvg_root: Path,
    lpips_metric: torch.nn.Module,
    dists_metric: torch.nn.Module,
    device: torch.device,
) -> dict:
    reports = []
    for sequence_name in UVG_SEQUENCES:
        frames_dir = (
            uvg_root
            / f"{sequence_name}_1920x1080_120fps_420_8bit_YUV_frames"
        )
        video, _ = load_png_sequence(frames_dir, UVG_FRAMES, start_frame=1)
        frame_offset = 0
        squared_error = 0.0
        value_count = 0
        lpips_sum = 0.0
        dists_sum = 0.0
        raw_bits = 0
        codec_ms = 0.0

        for frame_count in UVG_SEGMENTS:
            reference = video[frame_offset : frame_offset + frame_count].float().div(255.0)
            model_input = (
                reference.permute(1, 0, 2, 3).unsqueeze(0).mul(2.0).sub(1.0)
            )
            pad_info = None
            if tokenizer_name == "cosmos_cv488":
                model_input, pad_info = pad_for_cosmos(
                    model_input,
                    temporal_alignment=4,
                    spatial_alignment=16,
                )
            model_input = model_input.to(device)

            synchronize(device)
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latent = model.encode(model_input)
                quantized = model.quantize(latent)
                reconstruction = model.decode(quantized.decoder_latent)
            synchronize(device)
            codec_ms += (time.perf_counter() - started) * 1000.0

            if pad_info is not None:
                reconstruction = unpad_reconstruction(reconstruction, pad_info)
            reconstruction = (
                reconstruction.float().clamp(-1.0, 1.0).add(1.0).div(2.0)
                .squeeze(0).permute(1, 0, 2, 3).cpu()
            )
            difference = reference - reconstruction
            squared_error += float(difference.square().sum().item())
            value_count += difference.numel()
            lpips_sum += sequence_lpips(
                reference,
                reconstruction,
                lpips_metric,
                device=device,
                batch_size=1,
            ) * frame_count
            dists_sum += sequence_dists(
                reference, reconstruction, dists_metric, device
            ) * frame_count
            raw_bits += quantized.indices.numel() * 12
            frame_offset += frame_count

        pixels = frame_offset * video.shape[-2] * video.shape[-1]
        report = {
            "sequence": sequence_name,
            "frames": frame_offset,
            "psnr": -10.0 * math.log10(squared_error / value_count),
            "bpp": raw_bits / pixels,
            "lpips": lpips_sum / frame_offset,
            "dists": dists_sum / frame_offset,
            "latency_ms": codec_ms / frame_offset,
            "raw_bits": raw_bits,
            "squared_error": squared_error,
            "value_count": value_count,
        }
        reports.append(report)
        print(
            f"UVG {sequence_name}: bpp={report['bpp']:.8f} "
            f"PSNR={report['psnr']:.4f} LPIPS={report['lpips']:.6f} "
            f"DISTS={report['dists']:.6f}",
            flush=True,
        )

    total_frames = sum(report["frames"] for report in reports)
    total_pixels = total_frames * 1080 * 1920
    total_error = sum(report["squared_error"] for report in reports)
    total_values = sum(report["value_count"] for report in reports)
    return {
        "schema_version": 1,
        "dataset": "UVG",
        "tokenizer": tokenizer_name,
        "protocol": {
            "sequences": list(UVG_SEQUENCES),
            "frames_per_sequence": UVG_FRAMES,
            "segments": list(UVG_SEGMENTS),
            "rate": "12 raw bits per 4096-way index without entropy coding",
        },
        "aggregate": {
            "psnr": -10.0 * math.log10(total_error / total_values),
            "bpp": sum(report["raw_bits"] for report in reports) / total_pixels,
            "lpips": sum(report["lpips"] * report["frames"] for report in reports)
            / total_frames,
            "dists": sum(report["dists"] * report["frames"] for report in reports)
            / total_frames,
            "latency_ms": sum(
                report["latency_ms"] * report["frames"] for report in reports
            )
            / total_frames,
        },
        "sequences": reports,
    }



@torch.inference_mode()
def evaluate_uvg_v2(
    model: Stage1VQTokenizer,
    tokenizer_name: str,
    uvg_root: Path,
    lpips_metric: torch.nn.Module,
    dists_metric: torch.nn.Module,
    device: torch.device,
) -> dict:
    reports = []
    for sequence_name in UVG_SEQUENCES:
        frames_dir = (
            uvg_root
            / f"{sequence_name}_1920x1080_120fps_420_8bit_YUV_frames"
        )
        source_frames = uvg_source_frames(tokenizer_name)
        video, _ = load_png_sequence(frames_dir, source_frames, start_frame=1)
        prepared = prepare_uvg_input(video.float().div(255.0), tokenizer_name)
        synchronize(device)
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent = model.encode(prepared.model_input.to(device))
            quantized = model.quantize(latent)
            reconstruction = model.decode(quantized.decoder_latent)
        synchronize(device)
        codec_ms = (time.perf_counter() - started) * 1000.0

        reconstruction = finish_uvg_reconstruction(reconstruction, prepared)
        reference = prepared.reference
        difference = reference - reconstruction
        squared_error = float(difference.square().sum().item())
        value_count = difference.numel()
        raw_bits = quantized.indices.numel() * 12
        rate_frames = TOKENIZER_SPECS[tokenizer_name].input_frames
        rate_pixels = rate_frames * reference.shape[-2] * reference.shape[-1]
        report = {
            "sequence": sequence_name,
            "frames": reference.shape[0],
            "rate_frames": rate_frames,
            "resolution": [reference.shape[-2], reference.shape[-1]],
            "psnr": -10.0 * math.log10(squared_error / value_count),
            "bpp": raw_bits / rate_pixels,
            "lpips": sequence_lpips(
                reference, reconstruction, lpips_metric, device=device, batch_size=1
            ),
            "dists": sequence_dists(reference, reconstruction, dists_metric, device),
            "latency_ms": codec_ms / reference.shape[0],
            "raw_bits": raw_bits,
            "squared_error": squared_error,
            "value_count": value_count,
        }
        reports.append(report)
        print(
            f"UVG {sequence_name}: bpp={report['bpp']:.8f} "
            f"PSNR={report['psnr']:.4f} LPIPS={report['lpips']:.6f} "
            f"DISTS={report['dists']:.6f}",
            flush=True,
        )

    total_frames = sum(report["frames"] for report in reports)
    total_rate_pixels = sum(
        report["rate_frames"]
        * report["resolution"][0]
        * report["resolution"][1]
        for report in reports
    )
    total_error = sum(report["squared_error"] for report in reports)
    total_values = sum(report["value_count"] for report in reports)
    return {
        "schema_version": 2,
        "dataset": "UVG",
        "tokenizer": tokenizer_name,
        "protocol": {
            "sequences": list(UVG_SEQUENCES),
            "model_input_frames": TOKENIZER_SPECS[tokenizer_name].input_frames,
            "metric_frames": uvg_metric_frames(tokenizer_name),
            "bpp_denominator_frames": TOKENIZER_SPECS[tokenizer_name].input_frames,
            "rate": "12 raw bits per 4096-way index without entropy coding",
        },
        "aggregate": {
            "psnr": -10.0 * math.log10(total_error / total_values),
            "bpp": sum(report["raw_bits"] for report in reports)
            / total_rate_pixels,
            "lpips": sum(report["lpips"] * report["frames"] for report in reports)
            / total_frames,
            "dists": sum(report["dists"] * report["frames"] for report in reports)
            / total_frames,
            "latency_ms": sum(
                report["latency_ms"] * report["frames"] for report in reports
            )
            / total_frames,
        },
        "sequences": reports,
    }

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


def run(args: argparse.Namespace) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    spec = TOKENIZER_SPECS[args.tokenizer]
    config = load_stage1_training_config(args.config)
    output_dir = Path(args.output_dir)
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
        if resume_payload["config"] != asdict(config):
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
        lr=config.optimizer.learning_rate,
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
        "checkpoint_and_uvg_interval": CHECKPOINT_INTERVAL,
        "learning_rate_schedule": {
            "type": "step_decay",
            "interval_steps": config.optimizer.lr_decay_interval_steps,
            "factor": config.optimizer.lr_decay_factor,
            "stop_step": config.optimizer.lr_decay_stop_step,
        },
        "precision": "BF16 autocast with FP32 VQ assignment and losses",
        "world_size": world_size,
        "batch_size_per_gpu": config.train.batch_size,
        "global_batch_size": config.train.batch_size * world_size,
        "config": asdict(config),
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
        learning_rate = learning_rate_at_step(
            config.optimizer.learning_rate,
            step,
            interval_steps=config.optimizer.lr_decay_interval_steps,
            decay_factor=config.optimizer.lr_decay_factor,
            stop_step=config.optimizer.lr_decay_stop_step,
        )
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

        if step % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = output_dir / "checkpoints" / f"step_{step:07d}.pt"
            if rank == 0:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    args.tokenizer,
                    spec.input_frames,
                    spec.latent_channels,
                    spec.codebook_dim,
                    asdict(config),
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
                output_dir / "evaluations" / f"step_{step:07d}.json",
                model,
                optimizer,
                lpips_metric,
                rank=rank,
                local_rank=local_rank,
                device=device,
            )
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
