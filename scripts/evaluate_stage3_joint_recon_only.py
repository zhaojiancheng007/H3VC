from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.video_sequence import load_png_sequence
from entropy_model.model_categorical import CategoricalEntropyModel
from scripts.evaluate_stage3_joint_benchmark_recon import (
    dataset_sequences,
    save_reconstruction,
)
from scripts.stage3 import load_stage3_checkpoint
from tokenizer.stage3_vq import JointVQModel, selected_code_vectors
from tokenizer.stage_checkpoint import load_stage1_tokenizer
from training.config_stage3 import load_stage3_config, stage3_config_dict
from training.uvg_protocol import (
    finish_uvg_reconstruction,
    prepare_uvg_input,
    uvg_source_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=("uvg", "mcl_jcv", "hevc_b"))
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--sequence-start", required=True, type=int)
    parser.add_argument("--sequence-end", required=True, type=int)
    parser.add_argument("--wan-repo", default=str(PROJECT_ROOT / "Wan2.1"))
    parser.add_argument("--wan-checkpoint", default=str(PROJECT_ROOT / "pretrained_ckpts/Wan2.1/Wan2.1_VAE.pth"))
    parser.add_argument("--cosmos-repo", default=str(PROJECT_ROOT / "Cosmos-Tokenizer"))
    parser.add_argument("--cosmos-encoder", default=str(PROJECT_ROOT / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV4x8x8/encoder.jit"))
    parser.add_argument("--cosmos-decoder", default=str(PROJECT_ROOT / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV4x8x8/decoder.jit"))
    parser.add_argument("--cosmos-cv888-encoder", default=str(PROJECT_ROOT / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV8x8x8/encoder.jit"))
    parser.add_argument("--cosmos-cv888-decoder", default=str(PROJECT_ROOT / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV8x8x8/decoder.jit"))
    parser.add_argument("--minimax-h3-model-dir", default=str(PROJECT_ROOT / "pretrained_ckpts/MiniMax-H3/FL2VA/video_vae"))
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
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
    model = JointVQModel(loaded.model, CategoricalEntropyModel(config.model)).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)

    all_sequences = dataset_sequences(args.dataset, args.dataset_root)
    if not 1 <= args.sequence_start <= args.sequence_end <= len(all_sequences):
        raise ValueError(
            f"Invalid sequence range {args.sequence_start}:{args.sequence_end} "
            f"for {len(all_sequences)} sequences"
        )
    sequences = all_sequences[args.sequence_start - 1 : args.sequence_end]
    source_frames = uvg_source_frames(loaded.name)

    for sequence_name, frames_dir in sequences:
        sequence_output = args.output_dir / "recon" / sequence_name
        if sequence_output.exists():
            raise FileExistsError(sequence_output)
        video, _ = load_png_sequence(frames_dir, source_frames, start_frame=1)
        prepared = prepare_uvg_input(video.float().div(255.0), loaded.name)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent = model.tokenizer.encode(prepared.model_input.to(device))
            projected = model.tokenizer.pre_quant(latent)
        flat = projected.permute(0, 2, 3, 4, 1).reshape(-1, projected.shape[1]).float()
        flat_indices = model.tokenizer.quantizer._nearest_indices(flat)
        indices = flat_indices.reshape(
            projected.shape[0], projected.shape[2], projected.shape[3], projected.shape[4]
        )
        selected = selected_code_vectors(model.tokenizer.quantizer.embed, indices)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reconstruction = model.tokenizer.decode(
                model.tokenizer.post_quant(selected.to(projected.dtype))
            )
        reconstruction = finish_uvg_reconstruction(reconstruction, prepared)
        save_reconstruction(reconstruction, sequence_output)
        print(f"completed {sequence_name}", flush=True)


if __name__ == "__main__":
    main()
