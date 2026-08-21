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
from scripts.stage3 import load_stage3_checkpoint
from tokenizer.stage3_vq import JointVQModel
from tokenizer.stage_checkpoint import load_stage1_tokenizer
from training.config_stage3 import load_stage3_config, stage3_config_dict
from scripts.evaluate_stage2_entropy_mcl_jcv import evaluate_mcl_jcv
from training.video_losses import create_lpips_vgg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mcl-jcv-root",
        default="/luojix/data/MCL-JCV/figure4_bt709_first96",
    )
    parser.add_argument("--wan-repo", default=str(PROJECT_ROOT / "Wan2.1"))
    parser.add_argument("--wan-checkpoint", default=str(PROJECT_ROOT / "pretrained_ckpts/Wan2.1/Wan2.1_VAE.pth"))
    parser.add_argument("--cosmos-repo", default=str(PROJECT_ROOT / "Cosmos-Tokenizer"))
    parser.add_argument("--cosmos-encoder", default=str(PROJECT_ROOT / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV4x8x8/encoder.jit"))
    parser.add_argument("--cosmos-decoder", default=str(PROJECT_ROOT / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV4x8x8/decoder.jit"))
    parser.add_argument("--cosmos-cv888-encoder", default=str(PROJECT_ROOT / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV8x8x8/encoder.jit"))
    parser.add_argument("--cosmos-cv888-decoder", default=str(PROJECT_ROOT / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV8x8x8/decoder.jit"))
    parser.add_argument("--minimax-h3-model-dir", default=str(PROJECT_ROOT / "pretrained_ckpts/MiniMax-H3/FL2VA/video_vae"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda", 0)
    config = load_stage3_config(args.config)
    checkpoint = load_stage3_checkpoint(args.checkpoint)
    if checkpoint["config"] != stage3_config_dict(config):
        raise ValueError("Stage3 checkpoint and evaluation config differ")
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
    model.eval()
    evaluation = evaluate_mcl_jcv(
        model.tokenizer,
        model.entropy_model,
        loaded.name,
        Path(args.mcl_jcv_root),
        create_lpips_vgg(device),
        create_standard_lpips(device),
        DISTS().to(device).eval(),
        device,
    )
    evaluation["checkpoint_step"] = int(checkpoint["step"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, evaluation)


if __name__ == "__main__":
    main()
