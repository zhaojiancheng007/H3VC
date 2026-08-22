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
from scripts.stage3_cosine import load_stage3_checkpoint
from tokenizer.gaussian_tile_decode import (
    enable_gaussian_tile_decode,
    gaussian_tile_blend_protocol,
)
from tokenizer.stage3_vq import JointVQModel
from tokenizer.stage_checkpoint import load_stage1_tokenizer
from training.config_stage3_cosine import load_stage3_config, stage3_config_dict
from training.checkpoint_config_fast import assert_fast_loader_resume_config
from training.evaluation_categorical_distributed import (
    build_benchmark_jobs,
    evaluate_shard,
)
from training.video_losses import create_lpips_vgg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one distributed Stage3 Gaussian benchmark shard"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--recon-dir", required=True, type=Path)
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
    config = load_stage3_config(args.config)
    checkpoint = load_stage3_checkpoint(args.checkpoint)
    assert_fast_loader_resume_config(
        checkpoint["config"], stage3_config_dict(config), stage="Stage3"
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
    model = JointVQModel(
        loaded.model, CategoricalEntropyModel(config.model)
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    enable_gaussian_tile_decode(model.tokenizer, loaded.name)

    jobs = build_benchmark_jobs(
        args.uvg_root, args.mcl_jcv_root, args.hevc_b_root
    )
    evaluation = evaluate_shard(
        jobs,
        model.tokenizer,
        model.entropy_model,
        loaded.name,
        create_lpips_vgg(device),
        create_standard_lpips(device),
        DISTS().to(device).eval(),
        device,
        args.recon_dir,
        rank=args.rank,
        world_size=args.world_size,
    )
    evaluation["checkpoint_step"] = int(checkpoint["step"])
    evaluation["protocol"]["tile_blend"] = gaussian_tile_blend_protocol()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, evaluation)


if __name__ == "__main__":
    main()
