from __future__ import annotations

import argparse
import gc
import subprocess
import csv
from dataclasses import asdict
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import sys
import time


CPU_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
for variable_name in CPU_THREAD_ENVIRONMENT:
    os.environ[variable_name] = "1"


import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from entropy_model.config_local_global_memory import LocalGlobalMemoryEntropyConfig
from entropy_model.selection import normalize_stage2_checkpoint, build_model, config_from_checkpoint
from entropy_model.model_local_global_memory import LocalGlobalMemoryEntropyModel
from metrics.reporting import write_json
from training.periodic_benchmarks import evaluation_environment
from training.evaluation_categorical_distributed_codebook import merge_evaluation_shards, codebook_statistics
from tokenizer.stage3_vq import (
    JointTokenizerOutput,
    JointVQModel,
    JointVQOutput,
    relaxed_quantize,
)
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
from training.video_losses import chunked_video_lpips, create_lpips_vgg


CHECKPOINT_VERSION = 3
LPIPS_FRAME_BATCH_SIZE = 16
CPU_SAFE_PREFETCH_FACTOR = 2
EXPECTED_WORLD_SIZE = 8
EXPECTED_BATCH_SIZE_PER_GPU = 1


def _initialize_video_worker(_: int) -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def build_video_dataloader(
    dataset: Dataset[torch.Tensor],
    batch_sampler: ResumableDistributedBatchSampler,
    loader_generator: torch.Generator,
    *,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        generator=loader_generator,
        prefetch_factor=CPU_SAFE_PREFETCH_FACTOR,
        persistent_workers=True,
        worker_init_fn=_initialize_video_worker,
    )


class EntropyFloorJointVQModel(JointVQModel):
    """Expose differentiable assignments for the marginal-entropy constraint."""

    def __init__(self, tokenizer, entropy_model) -> None:
        super().__init__(tokenizer, entropy_model)
        self.target_probabilities: torch.Tensor | None = None

    def forward(self, video: torch.Tensor, *, temperature: float) -> JointVQOutput:
        latent = self.tokenizer.encode(video)
        projected = self.tokenizer.pre_quant(latent)
        relaxed = relaxed_quantize(
            projected,
            self.tokenizer.quantizer.embed,
            temperature=temperature,
        )
        reconstruction = self.tokenizer.decode(
            self.tokenizer.post_quant(relaxed.quantized.to(projected.dtype))
        )
        entropy = self.entropy_model.forward_soft(
            relaxed.quantized,
            relaxed.indices,
            relaxed.target_probabilities,
            k=1,
        )
        self.target_probabilities = relaxed.target_probabilities
        return JointVQOutput(
            tokenizer=JointTokenizerOutput(
                reconstruction=reconstruction,
                projected=relaxed.projected,
                selected=relaxed.selected,
                quantized=relaxed.quantized,
                indices=relaxed.indices,
            ),
            entropy=entropy,
        )


def load_in_two_rank_waves(load_fn, *args, **kwargs):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    loaded = None
    for first_rank in range(0, world_size, 2):
        if first_rank <= rank < min(first_rank + 2, world_size):
            loaded = load_fn(*args, **kwargs)
        dist.barrier()
    if loaded is None:
        raise RuntimeError(f"Rank {rank} did not enter a model-loading wave")
    return loaded


def global_marginal_entropy_bits(target_probabilities: torch.Tensor) -> torch.Tensor:
    local_marginal = target_probabilities.reshape(
        -1, target_probabilities.shape[-1]
    ).mean(dim=0)
    global_marginal = local_marginal.detach().clone()
    dist.all_reduce(global_marginal)
    global_marginal /= dist.get_world_size()
    differentiable_marginal = local_marginal + (
        global_marginal - local_marginal.detach()
    )
    return -(
        differentiable_marginal
        * torch.log2(differentiable_marginal.clamp_min(1e-30))
    ).sum()


def entropy_floor_penalty_bpp(
    marginal_entropy_bits: torch.Tensor,
    *,
    floor_bits: float,
    multiplier: float,
    latent_density: float,
) -> torch.Tensor:
    gap_bits = torch.relu(
        marginal_entropy_bits.new_tensor(floor_bits) - marginal_entropy_bits
    )
    return multiplier * gap_bits * latent_density


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Stage3 with a selected entropy model and a marginal-entropy floor"
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
    parser.add_argument("--resume")
    parser.add_argument("--init-checkpoint")
    parser.add_argument(
        "--allow-checkpoint-relocation",
        action="store_true",
        help="Accept explicitly supplied source checkpoints as relocated exact copies",
    )
    parser.add_argument(
        "--minimax-h3-model-dir",
        required=True,
    )
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--marginal-entropy-floor-bits", type=float, required=True)
    parser.add_argument("--entropy-floor-multiplier", type=float, required=True)
    return parser.parse_args()


def rate_parameters(step: int, config: Stage3Config) -> tuple[float, float]:
    return config.loss.rate_weight, config.loss.temperature


def estimated_bpp(estimated_bits: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
    pixels = video.shape[0] * video.shape[2] * video.shape[3] * video.shape[4]
    return estimated_bits.sum() / pixels


def load_stage2_checkpoint(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload = normalize_stage2_checkpoint(payload)
    required = {
        "architecture",
        "evaluation_k",
        "k",
        "mixed_prefix",
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
        "training_k_probabilities",
        "rng_states",
    }
    if set(payload) != required:
        raise ValueError(
            f"Stage2 checkpoint fields are {sorted(payload)}, "
            f"expected {sorted(required)}"
        )
    config_from_checkpoint(payload)
    if payload["stage"] != 2:
        raise ValueError("Expected Stage2 checkpoint")
    if payload["k"] != 1 or payload["evaluation_k"] != 1:
        raise ValueError("Stage3 requires Stage2 training and evaluation k=1")
    if payload["mixed_prefix"]:
        raise ValueError("Stage3 requires fixed k=1 rather than mixed-prefix training")
    return payload


def build_entropy_model(
    stage2: dict,
    stage3_model_config,
):
    model_config_payload = dict(stage2["config"]["model"])
    stage3_model_payload = asdict(stage3_model_config)
    shared_stage2_payload = {
        key: model_config_payload[key] for key in stage3_model_payload
    }
    if shared_stage2_payload != stage3_model_payload:
        raise ValueError("Stage2 and Stage3 shared entropy model configs differ")
    entropy_config = config_from_checkpoint(stage2)
    entropy_model = build_model(entropy_config)
    entropy_model.load_state_dict(stage2["model"], strict=True)
    return entropy_model, entropy_config


def load_stage3_checkpoint(path: str | Path, *, mmap: bool = False) -> dict:
    payload = torch.load(
        path, map_location="cpu", weights_only=False, mmap=mmap
    )
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


def rebuild_ddp_without_evaluation(
    model: JointVQModel, local_rank: int
) -> DistributedDataParallel:
    return DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )


def evaluate_saved_checkpoint(args, checkpoint_path, evaluation_dir, model, optimizer, lpips_metric, *, rank, local_rank, world_size, device):
    rng = capture_rng_state(device)
    model.to("cpu")
    lpips_metric.to("cpu")
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(PROJECT_ROOT / "scripts/evaluate.py"),
               "--checkpoint", str(checkpoint_path), "--tokenizer-checkpoint", args.tokenizer_checkpoint,
               "--entropy-checkpoint", args.entropy_checkpoint,
               "--output", str(evaluation_dir / f"rank_{rank:02d}.json"),
               "--recon-dir", str(evaluation_dir / "recon"), "--rank", str(rank),
               "--world-size", str(world_size), "--k", "1",
               "--uvg-root", args.uvg_root, "--mcl-jcv-root", args.mcl_jcv_root,
               "--hevc-b-root", args.hevc_b_root, "--minimax-h3-model-dir", args.minimax_h3_model_dir]
    dist.barrier()
    with (evaluation_dir / f"rank_{rank:02d}.log").open("w") as log:
        subprocess.run(command, cwd=PROJECT_ROOT, env=evaluation_environment(local_rank), stdout=log, stderr=subprocess.STDOUT, check=True)
    dist.barrier()
    if rank == 0:
        shards = [json.loads((evaluation_dir / f"rank_{i:02d}.json").read_text()) for i in range(world_size)]
        results = merge_evaluation_shards(shards)
        rows = []
        for name, result in results.items():
            counts = torch.tensor([item["codebook_histogram"] for item in result["sequences"]], dtype=torch.int64).sum(0)
            stats = codebook_statistics(counts)
            result["checkpoint"] = str(checkpoint_path)
            result["codebook"] = stats
            result["codebook_histogram"] = counts.tolist()
            result["estimated_bits_per_token"] = sum(item["bits"] for item in result["sequences"]) / int(counts.sum())
            write_json(evaluation_dir / f"{name}.json", result)
            rows.append({"dataset": name, "checkpoint_step": result["checkpoint_step"], **result["aggregate"], **stats,
                         "estimated_bits_per_token": result["estimated_bits_per_token"]})
        write_json(evaluation_dir / "summary.json", {"rows": rows, "protocol": shards[0]["protocol"]})
        with (evaluation_dir / "summary.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    dist.barrier()
    model.to(device)
    lpips_metric.to(device)
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    torch.cuda.empty_cache()
    restore_rng_state(rng, device)
    return rebuild_ddp_without_evaluation(model, local_rank)


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
    if world_size != EXPECTED_WORLD_SIZE:
        raise ValueError(
            f"stage3.py requires {EXPECTED_WORLD_SIZE} GPUs, "
            f"got WORLD_SIZE={world_size}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    config = load_stage3_config(args.config)
    if config.train.batch_size != EXPECTED_BATCH_SIZE_PER_GPU:
        raise ValueError(
            "stage3.py requires batch_size=1 per GPU, "
            f"got {config.train.batch_size}"
        )
    config_payload = stage3_config_dict(config)
    loaded = load_in_two_rank_waves(
        load_stage1_tokenizer,
        args.tokenizer_checkpoint,

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
    stage2 = load_in_two_rank_waves(load_stage2_checkpoint, entropy_checkpoint)
    if stage2["tokenizer"] != loaded.name:
        raise ValueError("Stage1 and Stage2 tokenizer types differ")
    stage2_source_relocated = assert_checkpoint_reference(
        stage2["tokenizer_checkpoint"],
        loaded.checkpoint,
        label="Stage2 source Stage1 checkpoint",
        allow_relocation=args.allow_checkpoint_relocation,
    )
    entropy_model, entropy_config = build_entropy_model(stage2, config.model)
    model = EntropyFloorJointVQModel(loaded.model, entropy_model)
    initialization_relocated = False
    initialization = (
        None if args.init_checkpoint is None
        else load_in_two_rank_waves(load_stage3_checkpoint, args.init_checkpoint)
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
    resume = (
        None
        if args.resume is None
        else load_in_two_rank_waves(load_stage3_checkpoint, args.resume)
    )
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
                "entropy_architecture": stage2["architecture"],
                "entropy_model_config": asdict(entropy_config),
                "codebook": [4096, loaded.codebook_dim],
                "codebook_frozen": True,
                "world_size": world_size,
                "batch_size_per_gpu": config.train.batch_size,
                "global_batch_size": global_batch_size,
                "precision": "BF16 autocast with FP32 parameters, VQ assignment, and losses",
                "learning_rate_scheduler": "cosine_annealing",
                "rate_weight": config.loss.rate_weight,
                "temperature": config.loss.temperature,
                "anti_collapse": {
                    "method": "global_soft_marginal_entropy_floor",
                    "marginal_entropy_floor_bits": args.marginal_entropy_floor_bits,
                    "entropy_floor_multiplier": args.entropy_floor_multiplier,
                    "target_selection_gradient": "enabled",
                    "entropy_context_k": 1,
                    "periodic_evaluation": "standard_1080p_all42_every_checkpoint",
                },
                "startup_model_loading": "two_ranks_per_wave",
                "discriminator_training": "disabled",
                "checkpoint_relocation_allowed": args.allow_checkpoint_relocation,
                "checkpoint_relocated": (
                    stage2_source_relocated
                    or initialization_relocated
                    or resume_relocated
                ),
                "data_loader": {
                    "implementation": "opencv_worker_resize_crop",
                    "workers_per_rank": config.train.num_workers,
                    "prefetch_factor": CPU_SAFE_PREFETCH_FACTOR,
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
        if model.target_probabilities is None:
            raise RuntimeError("Joint model did not expose target probabilities")
        marginal_entropy_bits = global_marginal_entropy_bits(
            model.target_probabilities
        )
        conditional_bits_per_index = (
            output.entropy.soft_estimated_bits.sum()
            / output.tokenizer.indices.numel()
        )
        contextual_gain_bits = marginal_entropy_bits - conditional_bits_per_index
        pixels = video.shape[0] * video.shape[2] * video.shape[3] * video.shape[4]
        latent_density = output.tokenizer.indices.numel() / pixels
        entropy_floor_gap_bits = torch.relu(
            marginal_entropy_bits.new_tensor(args.marginal_entropy_floor_bits)
            - marginal_entropy_bits
        )
        entropy_floor_bpp = entropy_floor_penalty_bpp(
            marginal_entropy_bits,
            floor_bits=args.marginal_entropy_floor_bits,
            multiplier=args.entropy_floor_multiplier,
            latent_density=latent_density,
        )
        constrained_soft_bpp = soft_bpp + entropy_floor_bpp
        total_loss = (
            config.loss.mse_weight * mse_loss
            + config.loss.lpips_weight * lpips_loss.float()
            + config.loss.commitment_weight * commitment_loss
            + rate_weight * constrained_soft_bpp
        )
        total_loss.backward()
        model.target_probabilities = None
        if step == start_step + 1:
            group_gradients = {}
            for name, parameters in [("tokenizer", tokenizer_parameters), ("entropy_model", entropy_parameters)]:
                squared_norm = sum(parameter.grad.detach().float().square().sum() for parameter in parameters if parameter.grad is not None)
                if not torch.isfinite(squared_norm) or squared_norm <= 0:
                    raise RuntimeError(f"Invalid first-step {name} gradient: {squared_norm}")
                group_gradients[name] = float(squared_norm.sqrt())
            if model.tokenizer.quantizer.embed.requires_grad:
                raise RuntimeError("Codebook must remain frozen")
            if rank == 0:
                write_json(output_dir / f"gradient_check_step_{step:07d}.json", group_gradients)
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
                    marginal_entropy_bits.detach(),
                    conditional_bits_per_index.detach(),
                    contextual_gain_bits.detach(),
                    entropy_floor_bpp.detach(),
                    constrained_soft_bpp.detach(),
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
                "entropy_floor": float(scalars[11]),
            },
            "rate_weight": rate_weight,
            "temperature": temperature,
            "rate_objective_bpp": float(scalars[12]),
            "information": {
                "marginal_entropy_bits": float(scalars[8]),
                "conditional_bits_per_index": float(scalars[9]),
                "contextual_gain_bits": float(scalars[10]),
                "entropy_floor_gap_bits": max(
                    0.0, args.marginal_entropy_floor_bits - float(scalars[8])
                ),
                "entropy_floor_active": (
                    float(scalars[8]) < args.marginal_entropy_floor_bits
                ),
            },
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
                    f"H={metrics['information']['marginal_entropy_bits']:.4f} "
                    f"G={metrics['information']['contextual_gain_bits']:.4f} "
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
            del marginal_entropy_bits, conditional_bits_per_index
            del contextual_gain_bits, entropy_floor_gap_bits
            del entropy_floor_bpp, constrained_soft_bpp
            del histogram, probabilities, active, perplexity, grad_norm, psnr
            del ddp_model
            ddp_model = evaluate_saved_checkpoint(
                args, checkpoint_path, output_dir / "evaluations" / f"step_{step:07d}",
                model, optimizer, lpips_metric, rank=rank, local_rank=local_rank,
                world_size=world_size, device=device,
            )
            model.train()
            lpips_metric.eval()

    if progress is not None:
        progress.close()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.marginal_entropy_floor_bits <= 12.0:
        raise ValueError("marginal entropy floor must be in (0, 12] bits")
    if args.entropy_floor_multiplier <= 0.0:
        raise ValueError("entropy floor multiplier must be positive")
    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError("Use either --resume or --init-checkpoint, not both")
    dist.init_process_group("nccl", timeout=timedelta(hours=6))
    try:
        run(args)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
