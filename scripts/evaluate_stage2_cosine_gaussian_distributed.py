from __future__ import annotations

import argparse
from pathlib import Path
import sys

from DISTS_pytorch import DISTS
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from entropy_model.model_categorical import CategoricalEntropyModel
from metrics.lpips import create_standard_lpips
from metrics.reporting import write_json
from scripts.stage2_cosine import load_checkpoint
from tokenizer.gaussian_tile_decode import (
    enable_gaussian_tile_decode,
    gaussian_tile_blend_protocol,
)
from tokenizer.stage_checkpoint import load_stage1_tokenizer
from training.config_stage2_cosine import load_stage2_config, stage2_config_dict
from training.checkpoint_config_fast import assert_fast_loader_resume_config
from training.evaluation_categorical_distributed import (
    build_benchmark_jobs,
    evaluate_shard,
)
from training.video_losses import create_lpips_vgg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one shard of all Stage2 benchmarks on one GPU"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--uvg-root", required=True)
    parser.add_argument("--mcl-jcv-root", required=True)
    parser.add_argument("--hevc-b-root", required=True)
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
        default=str(
            PROJECT_ROOT
            / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV8x8x8/encoder.jit"
        ),
    )
    parser.add_argument(
        "--cosmos-cv888-decoder",
        default=str(
            PROJECT_ROOT
            / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV8x8x8/decoder.jit"
        ),
    )
    parser.add_argument(
        "--minimax-h3-model-dir",
        default=str(PROJECT_ROOT / "pretrained_ckpts/MiniMax-H3/FL2VA/video_vae"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    config = load_stage2_config(args.config)
    checkpoint = load_checkpoint(args.checkpoint)
    assert_fast_loader_resume_config(
        checkpoint["config"], stage2_config_dict(config), stage="Stage2"
    )

    loaded = load_stage1_tokenizer(
        checkpoint["tokenizer_checkpoint"],
        wan_repo=args.wan_repo,
        wan_checkpoint=args.wan_checkpoint,
        cosmos_repo=args.cosmos_repo,
        cosmos_encoder=args.cosmos_encoder,
        cosmos_decoder=args.cosmos_decoder,
        cosmos_cv888_encoder=args.cosmos_cv888_encoder,
        cosmos_cv888_decoder=args.cosmos_cv888_decoder,
        minimax_h3_model_dir=args.minimax_h3_model_dir,
    )
    tokenizer = loaded.model.to(device).eval().requires_grad_(False)
    enable_gaussian_tile_decode(tokenizer, loaded.name)
    entropy_model = CategoricalEntropyModel(config.model).to(device).eval()
    entropy_model.load_state_dict(checkpoint["model"], strict=True)
    lpips_metric = create_lpips_vgg(device)
    lpips_alex_metric = create_standard_lpips(device)
    dists_metric = DISTS().to(device).eval()

    jobs = build_benchmark_jobs(
        args.uvg_root, args.mcl_jcv_root, args.hevc_b_root
    )
    shard = evaluate_shard(
        jobs,
        tokenizer,
        entropy_model,
        loaded.name,
        lpips_metric,
        lpips_alex_metric,
        dists_metric,
        device,
        None,
        rank=args.rank,
        world_size=args.world_size,
    )
    shard["checkpoint_step"] = int(checkpoint["step"])
    shard["protocol"]["tile_blend"] = gaussian_tile_blend_protocol()
    write_json(args.output, shard)


if __name__ == "__main__":
    main()
