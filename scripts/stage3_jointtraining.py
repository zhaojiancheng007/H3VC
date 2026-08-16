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
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from entropy_model.model_categorical import CategoricalEntropyModel
from metrics.reporting import write_json
from tokenizer.stage3_vq import JointVQModel
from tokenizer.stage_checkpoint import load_stage1_tokenizer
from training.checkpoint_ar_ddp import DDPTrainingState
from training.config_stage3_joint import Stage3Config, load_stage3_config
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
    capture_rng_state,
    collective_error,
    rank_zero_output_setup,
    restore_rng_state,
)
from training.gan_stage3 import (
    PatchDiscriminator,
    discriminator_microbatch_backward,
    generator_adversarial_loss,
    select_gan_frames,
)
from training.video_losses import create_lpips_vgg, video_lpips


CHECKPOINT_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jointly train a Stage1 VQ tokenizer and Stage2 entropy model"
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/stage3_joint_vq.yaml"))
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument(
        "--dataset-format",
        default=DEFAULT_DATASET_FORMAT,
        choices=("openvid", "trainable_jsons"),
    )
    parser.add_argument("--tokenizer-checkpoint", required=True)
    parser.add_argument("--entropy-checkpoint", required=True)
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
        default=str(PROJECT_ROOT / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV4x8x8/encoder.jit"),
    )
    parser.add_argument(
        "--cosmos-decoder",
        default=str(PROJECT_ROOT / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV4x8x8/decoder.jit"),
    )
    parser.add_argument("--resume")
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
    parser.add_argument("--stop-after-steps", type=int)
    return parser.parse_args()


def rate_parameters(step: int, config: Stage3Config) -> tuple[float, float]:
    loss = config.loss
    if step <= loss.rate_warmup_end_step:
        progress = step / loss.rate_warmup_end_step
        rate_weight = loss.initial_rate_weight + progress * (
            loss.base_rate_weight - loss.initial_rate_weight
        )
        temperature = loss.initial_temperature
    elif step <= loss.rate_ramp_start_step:
        rate_weight = loss.base_rate_weight
        temperature = loss.initial_temperature
    elif step <= loss.rate_ramp_end_step:
        progress = (step - loss.rate_ramp_start_step) / (
            loss.rate_ramp_end_step - loss.rate_ramp_start_step
        )
        rate_weight = loss.base_rate_weight + progress * (
            loss.final_rate_weight - loss.base_rate_weight
        )
        temperature = loss.final_temperature
    else:
        rate_weight = loss.final_rate_weight
        temperature = loss.final_temperature
    return rate_weight, temperature


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
    if payload["version"] != 1 or payload["stage"] != 2:
        raise ValueError("Expected a version-2 Stage2 checkpoint")
    return payload


def load_stage3_checkpoint(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "version",
        "stage",
        "step",
        "tokenizer",
        "tokenizer_checkpoint",
        "entropy_checkpoint",
        "config",
        "model",
        "optimizer",
        "discriminator",
        "discriminator_optimizer",
        "training_state",
        "rng_states",
    }
    if set(payload) != required:
        raise ValueError(
            f"Stage3 checkpoint fields are {sorted(payload)}, "
            f"expected {sorted(required)}"
        )
    if payload["version"] != CHECKPOINT_VERSION or payload["stage"] != 3:
        raise ValueError("Expected a version-2 Stage3 checkpoint")
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
    evaluation_path: Path,
    model: JointVQModel,
    optimizer: torch.optim.Optimizer,
    discriminator: PatchDiscriminator,
    discriminator_optimizer: torch.optim.Optimizer,
    lpips_metric: torch.nn.Module,
    *,
    rank: int,
    local_rank: int,
    device: torch.device,
) -> tuple[DistributedDataParallel, DistributedDataParallel]:
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    if rank == 0:
        model.to("cpu")
        discriminator.to("cpu")
        move_optimizer_state(optimizer, torch.device("cpu"))
        move_optimizer_state(discriminator_optimizer, torch.device("cpu"))
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
                str(PROJECT_ROOT / "scripts/evaluate_stage3_joint_uvg_v2.py"),
                "--checkpoint",
                str(checkpoint_path.resolve()),
                "--config",
                str(Path(args.config).resolve()),
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
        discriminator.to(device)
        move_optimizer_state(optimizer, device)
        move_optimizer_state(discriminator_optimizer, device)
        lpips_metric.to(device)
        torch.cuda.empty_cache()

    dist.barrier()
    ddp_model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    ddp_discriminator = DistributedDataParallel(
        discriminator,
        device_ids=[local_rank],
        broadcast_buffers=True,
        find_unused_parameters=False,
    )
    return ddp_model, ddp_discriminator


def save_checkpoint(
    path: Path,
    model: JointVQModel,
    optimizer: torch.optim.Optimizer,
    discriminator: PatchDiscriminator,
    discriminator_optimizer: torch.optim.Optimizer,
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
            "config": asdict(config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "discriminator": discriminator.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
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
    config = load_stage3_config(args.config)
    if config.train.bf16:
        raise ValueError("Stage3 requires FP32; set train.bf16 to false")
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

    entropy_checkpoint = Path(args.entropy_checkpoint).resolve()
    stage2 = load_stage2_checkpoint(entropy_checkpoint)
    if stage2["tokenizer"] != loaded.name:
        raise ValueError("Stage1 and Stage2 tokenizer types differ")
    if Path(stage2["tokenizer_checkpoint"]) != loaded.checkpoint:
        raise ValueError("Stage2 was trained with a different Stage1 checkpoint")
    if stage2["config"]["model"] != asdict(config.model):
        raise ValueError("Stage2 and Stage3 entropy model configs differ")

    entropy_model = CategoricalEntropyModel(config.model)
    entropy_model.load_state_dict(stage2["model"], strict=True)
    model = JointVQModel(loaded.model, entropy_model)
    discriminator = PatchDiscriminator(base_channels=64, num_layers=3)
    resume = None if args.resume is None else load_stage3_checkpoint(args.resume)
    start_step = 0
    start_epoch = 0
    start_batch = 0
    if resume is not None:
        if resume["config"] != asdict(config):
            raise ValueError("Stage3 resume config changed")
        if resume["tokenizer"] != loaded.name:
            raise ValueError("Stage3 resume tokenizer changed")
        state = DDPTrainingState(**resume["training_state"])
        if state.world_size != world_size or state.global_batch_size != global_batch_size:
            raise ValueError("Stage3 resume distributed batch changed")
        model.load_state_dict(resume["model"], strict=True)
        discriminator.load_state_dict(resume["discriminator"], strict=True)
        start_step = state.step
        start_epoch = state.epoch
        start_batch = state.batch_in_epoch

    model = model.to(device)
    discriminator = discriminator.to(device)
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
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.optimizer.discriminator_learning_rate,
        betas=config.optimizer.betas,
    )
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
        discriminator_optimizer.load_state_dict(resume["discriminator_optimizer"])

    ddp_model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    ddp_discriminator = DistributedDataParallel(
        discriminator,
        device_ids=[local_rank],
        broadcast_buffers=True,
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
                "precision": "FP32",
                "config": asdict(config),
            },
        )
    metrics_path = output_dir / "train_metrics.jsonl"
    log_path = output_dir / "training.log"
    progress = tqdm(
        total=stop_step,
        initial=start_step,
        desc=f"Stage3 {loaded.name} joint",
        dynamic_ncols=True,
    ) if rank == 0 else None
    lpips_metric = create_lpips_vgg(device)
    trainable_parameters = [*tokenizer_parameters, *entropy_parameters]
    model.train()
    discriminator.train()

    for step in range(start_step + 1, stop_step + 1):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        video = stream.next().to(device, non_blocking=True)
        rate_weight, temperature = rate_parameters(step, config)
        gan_active = (
            config.loss.gan_weight > 0 and step >= config.loss.gan_start_step
        )
        optimizer.zero_grad(set_to_none=True)
        discriminator_optimizer.zero_grad(set_to_none=True)

        output = ddp_model(video, temperature=temperature)
        lpips_loss = video_lpips(
            video, output.tokenizer.reconstruction, lpips_metric
        )
        generator_gan_loss = torch.zeros((), device=device)
        if gan_active:
            real_frames, fake_frames = select_gan_frames(
                video, output.tokenizer.reconstruction
            )
            generator_gan_loss = generator_adversarial_loss(
                discriminator, fake_frames
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
            + config.loss.gan_weight * generator_gan_loss.float()
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

        discriminator_loss = torch.zeros((), device=device)
        discriminator_grad_norm = torch.zeros((), device=device)
        if gan_active:
            discriminator_loss = discriminator_microbatch_backward(
                ddp_discriminator,
                real_frames,
                fake_frames,
                loss_weight=config.loss.gan_weight,
                gradient_accumulation=1,
            )
            discriminator_grad_norm = torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(), config.optimizer.grad_clip_norm
            )
            collective_error(
                None
                if torch.isfinite(discriminator_grad_norm)
                else f"non-finite discriminator gradient at step {step}",
                device,
            )
            discriminator_optimizer.step()

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
                    generator_gan_loss.detach().float(),
                    discriminator_loss.detach().float(),
                    grad_norm.detach().float(),
                    discriminator_grad_norm.detach().float(),
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
                "generator_gan": float(scalars[7]),
                "discriminator": float(scalars[8]),
            },
            "rate_weight": rate_weight,
            "temperature": temperature,
            "gan_active": gan_active,
            "codebook": {
                "active": int(active.sum()),
                "perplexity": float(perplexity),
            },
            "grad_norm": float(scalars[9]),
            "discriminator_grad_norm": float(scalars[10]),
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
                    f"gan={gan_active} latency_ms={metrics['latency_ms']:.3f}",
                )

        if step % config.train.checkpoint_interval == 0:
            checkpoint_path = output_dir / "checkpoints" / f"step_{step:07d}.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                discriminator,
                discriminator_optimizer,
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
            discriminator_optimizer.zero_grad(set_to_none=True)
            del output, video, lpips_loss, mse_loss, commitment_loss
            del soft_bpp, hard_bpp, total_loss, generator_gan_loss
            del discriminator_loss, discriminator_grad_norm, scalars, latency_ms
            del histogram, probabilities, active, perplexity, grad_norm, psnr
            if gan_active:
                del real_frames, fake_frames
            del ddp_model, ddp_discriminator
            ddp_model, ddp_discriminator = evaluate_checkpoint_in_subprocess(
                args,
                checkpoint_path,
                output_dir / "evaluations" / f"step_{step:07d}.json",
                model,
                optimizer,
                discriminator,
                discriminator_optimizer,
                lpips_metric,
                rank=rank,
                local_rank=local_rank,
                device=device,
            )
            model.train()
            discriminator.train()
            lpips_metric.eval()

    save_checkpoint(
        output_dir / "checkpoints" / "last.pt",
        model,
        optimizer,
        discriminator,
        discriminator_optimizer,
        config,
        loaded.name,
        loaded.checkpoint,
        entropy_checkpoint,
        step=stop_step,
        stream=stream,
        world_size=world_size,
        global_batch_size=global_batch_size,
        rank=rank,
        device=device,
    )
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
