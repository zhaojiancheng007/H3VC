from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


for variable_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[variable_name] = "1"


from DISTS_pytorch import DISTS
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from entropy_model.model_local_global_memory import LocalGlobalMemoryEntropyModel
from entropy_model.config_local_global_memory import LocalGlobalMemoryEntropyConfig
from metrics.lpips import create_standard_lpips
from metrics.reporting import write_json
from scripts.stage3 import load_stage3_checkpoint
from entropy_model.selection import build_model, config_from_checkpoint, architecture
from tokenizer.gaussian_tile_decode import (
    enable_gaussian_tile_decode,
    gaussian_tile_blend_protocol,
)
from tokenizer.stage3_vq import JointVQModel
from tokenizer.stage_checkpoint import load_stage1_tokenizer
from training.evaluation_categorical_distributed_codebook import (
    build_benchmark_jobs,
    evaluate_shard,
)
from training.video_losses import create_lpips_vgg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one distributed Stage3 Gaussian benchmark shard"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--entropy-checkpoint", required=True)
    parser.add_argument("--tokenizer-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--recon-dir", required=True, type=Path)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--uvg-root", required=True)
    parser.add_argument("--mcl-jcv-root", required=True)
    parser.add_argument("--hevc-b-root", required=True)
    parser.add_argument(
        "--minimax-h3-model-dir",
        default=str(PROJECT_ROOT / "pretrained_ckpts/MiniMax-H3/FL2VA/video_vae"),
    )
    return parser.parse_args()


def load_model(args):
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    checkpoint = load_stage3_checkpoint(args.checkpoint, mmap=True)
    stage2 = torch.load(args.entropy_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model_config = config_from_checkpoint(stage2)
    entropy_architecture = architecture(model_config)
    del stage2

    loaded = load_stage1_tokenizer(
        args.tokenizer_checkpoint,

        minimax_h3_model_dir=args.minimax_h3_model_dir,
    )
    if loaded.name != checkpoint["tokenizer"]:
        raise ValueError("Stage1 tokenizer does not match the Stage3 checkpoint")
    model = JointVQModel(
        loaded.model, build_model(model_config)
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    checkpoint_step = int(checkpoint["step"])
    del checkpoint
    enable_gaussian_tile_decode(model.tokenizer, loaded.name)

    return model, loaded, checkpoint_step, entropy_architecture, device


def main() -> None:
    args = parse_args()
    model, loaded, checkpoint_step, entropy_architecture, device = load_model(args)
    jobs = build_benchmark_jobs(
        args.uvg_root, args.mcl_jcv_root, args.hevc_b_root
    )
    lpips_vgg_metric = create_lpips_vgg(device)
    lpips_alex_metric = create_standard_lpips(device)
    dists_metric = DISTS().to(device).eval()
    print(
        f"model_loaded rank={args.rank} checkpoint_step={checkpoint_step}",
        flush=True,
    )
    evaluation = evaluate_shard(
        jobs,
        model.tokenizer,
        model.entropy_model,
        loaded.name,
        lpips_vgg_metric,
        lpips_alex_metric,
        dists_metric,
        device,
        args.recon_dir,
        k=args.k,
        rank=args.rank,
        world_size=args.world_size,
    )
    evaluation["checkpoint"] = str(Path(args.checkpoint).resolve())
    evaluation["tokenizer_checkpoint"] = str(Path(args.tokenizer_checkpoint).resolve())
    evaluation["checkpoint_step"] = checkpoint_step
    evaluation["protocol"]["entropy_architecture"] = entropy_architecture
    evaluation["protocol"]["codebook_statistics"] = "hard assignments; all model latent indices; pool histograms across sequences"
    evaluation["protocol"]["tile_blend"] = gaussian_tile_blend_protocol()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, evaluation)


if __name__ == "__main__":
    main()
