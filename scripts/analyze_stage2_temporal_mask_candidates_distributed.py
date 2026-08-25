from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.openvid_mp4_sequence import TrainingSampleKey
from datasets.trainable_jsons_sequence_fast import SeededFastTrainableJsonsVideoDataset
from entropy_model.model_categorical_context_corruption import (
    CategoricalEntropyModelContextCorruption,
)
from metrics.reporting import write_json
from scripts.analyze_stage2_temporal_attention import _quantize_video
from scripts.stage2_cosine_context_corruption import load_checkpoint
from tokenizer.stage_backbones import TOKENIZER_SPECS
from tokenizer.stage_checkpoint import load_stage1_tokenizer
from training.checkpoint_config_fast import assert_fast_loader_resume_config
from training.config_stage2_context_corruption import (
    load_stage2_context_corruption_config,
    stage2_context_corruption_config_dict,
)
from training.stage_data_fast import DEFAULT_DATASET_ROOT, DEFAULT_METADATA
from training.temporal_attention_mask_candidates import (
    CANDIDATE_NAMES,
    candidate_layer_specs,
    evaluate_temporal_mask_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic temporal causal mask candidates"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--sample-indices", required=True, type=int, nargs="+")
    parser.add_argument("--crop-seed", type=int, default=2026)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assigned = args.sample_indices[args.rank :: args.world_size]
    if not assigned:
        raise ValueError(
            f"Rank {args.rank} has no sample from {args.sample_indices} at world size {args.world_size}"
        )
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    config = load_stage2_context_corruption_config(args.config)
    checkpoint = load_checkpoint(args.checkpoint)
    assert_fast_loader_resume_config(
        checkpoint["config"],
        stage2_context_corruption_config_dict(config),
        stage="Stage2 temporal-mask candidate diagnostic",
    )
    dataset = SeededFastTrainableJsonsVideoDataset(
        args.dataset_root,
        args.metadata,
        num_frames=TOKENIZER_SPECS[checkpoint["tokenizer"]].input_frames,
        rejection_log=None,
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
    model = CategoricalEntropyModelContextCorruption(config.model).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)

    samples = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for sample_index in assigned:
            video = dataset[TrainingSampleKey(sample_index, args.crop_seed)]
            selected, indices = _quantize_video(tokenizer, video, device)
            evaluations = evaluate_temporal_mask_candidates(
                model,
                selected,
                indices,
                rate_pixels=video.shape[1] * video.shape[2] * video.shape[3],
                device=device,
            )
            samples.append(
                {
                    "sample_index": sample_index,
                    "crop_seed": args.crop_seed,
                    "video_shape": list(video.shape),
                    "index_shape": list(indices.shape),
                    "candidates": evaluations,
                }
            )
    write_json(
        args.output,
        {
            "schema_version": 1,
            "rank": args.rank,
            "world_size": args.world_size,
            "checkpoint_step": int(checkpoint["step"]),
            "sample_indices": assigned,
            "candidate_order": list(CANDIDATE_NAMES),
            "layer_specs": {
                name: candidate_layer_specs(name, config.model.num_layers)
                for name in CANDIDATE_NAMES
            },
            "samples": samples,
        },
    )
    print(json.dumps({"rank": args.rank, "samples": assigned}, indent=2))


if __name__ == "__main__":
    main()
