from __future__ import annotations

import argparse
from pathlib import Path
import sys

from DISTS_pytorch import DISTS
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.lpips import create_standard_lpips
from metrics.reporting import write_json
from tokenizer.stage_checkpoint import load_stage1_tokenizer
from training.evaluation_categorical import evaluate_benchmark
from training.video_losses import create_lpips_vgg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one Stage1 tokenizer checkpoint on a shared benchmark"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True, choices=("uvg", "mcl_jcv", "hevc_b"))
    parser.add_argument("--dataset-root", required=True)
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
    parser.add_argument("--cosmos-cv888-encoder", default=str(PROJECT_ROOT / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV8x8x8/encoder.jit"))
    parser.add_argument("--cosmos-cv888-decoder", default=str(PROJECT_ROOT / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV8x8x8/decoder.jit"))
    parser.add_argument("--minimax-h3-model-dir", default=str(PROJECT_ROOT / "pretrained_ckpts/MiniMax-H3/FL2VA/video_vae"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    loaded = load_stage1_tokenizer(
        args.checkpoint,
        wan_repo=args.wan_repo,
        wan_checkpoint=args.wan_checkpoint,
        cosmos_repo=args.cosmos_repo,
        cosmos_encoder=args.cosmos_encoder,
        cosmos_decoder=args.cosmos_decoder,
        cosmos_cv888_encoder=args.cosmos_cv888_encoder,
        cosmos_cv888_decoder=args.cosmos_cv888_decoder,
        minimax_h3_model_dir=args.minimax_h3_model_dir,
    )
    model = loaded.model.to(device).eval()
    lpips_metric = create_lpips_vgg(device)
    lpips_alex_metric = create_standard_lpips(device)
    dists_metric = DISTS().to(device).eval()

    evaluation = evaluate_benchmark(
        model,
        None,
        loaded.name,
        args.dataset,
        Path(args.dataset_root),
        lpips_metric,
        lpips_alex_metric,
        dists_metric,
        device,
    )
    evaluation["checkpoint_step"] = loaded.step
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, evaluation)


if __name__ == "__main__":
    main()
