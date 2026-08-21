from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

from DISTS_pytorch import DISTS
import numpy as np
from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.video_sequence import load_png_sequence
from entropy_model.model_categorical import CategoricalEntropyModel
from metrics.lpips import create_standard_lpips, sequence_lpips
from metrics.reporting import write_json
from scripts.stage3 import load_stage3_checkpoint
from tokenizer.stage3_vq import JointVQModel, selected_code_vectors
from tokenizer.stage_backbones import TOKENIZER_SPECS
from tokenizer.stage_checkpoint import load_stage1_tokenizer
from training.config_stage3 import load_stage3_config, stage3_config_dict
from training.evaluation_categorical import (
    DATASET_NAMES,
    HEVC_B_SEQUENCES,
    MCL_JCV_SEQUENCES,
    UVG_SEQUENCES,
    _estimated_bits,
    _sequence_dists,
)
from training.uvg_protocol import (
    finish_uvg_reconstruction,
    prepare_uvg_input,
    uvg_metric_frames,
    uvg_source_frames,
)
from training.video_losses import create_lpips_vgg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=("uvg", "mcl_jcv", "hevc_b"))
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--wan-repo", default=str(PROJECT_ROOT / "Wan2.1"))
    parser.add_argument("--wan-checkpoint", default=str(PROJECT_ROOT / "pretrained_ckpts/Wan2.1/Wan2.1_VAE.pth"))
    parser.add_argument("--cosmos-repo", default=str(PROJECT_ROOT / "Cosmos-Tokenizer"))
    parser.add_argument("--cosmos-encoder", default=str(PROJECT_ROOT / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV4x8x8/encoder.jit"))
    parser.add_argument("--cosmos-decoder", default=str(PROJECT_ROOT / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV4x8x8/decoder.jit"))
    parser.add_argument("--cosmos-cv888-encoder", default=str(PROJECT_ROOT / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV8x8x8/encoder.jit"))
    parser.add_argument("--cosmos-cv888-decoder", default=str(PROJECT_ROOT / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV8x8x8/decoder.jit"))
    parser.add_argument("--minimax-h3-model-dir", default=str(PROJECT_ROOT / "pretrained_ckpts/MiniMax-H3/FL2VA/video_vae"))
    return parser.parse_args()


def dataset_sequences(dataset: str, root: Path) -> tuple[tuple[str, Path], ...]:
    if dataset == "uvg":
        return tuple(
            (name, root / f"{name}_1920x1080_120fps_420_8bit_YUV_frames")
            for name in UVG_SEQUENCES
        )
    names = {"mcl_jcv": MCL_JCV_SEQUENCES, "hevc_b": HEVC_B_SEQUENCES}[dataset]
    return tuple((name, root / name) for name in names)


def save_reconstruction(reconstruction: torch.Tensor, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = reconstruction.mul(255.0).round().to(torch.uint8)
    for frame_index, frame in enumerate(frames):
        pixels = frame.permute(1, 2, 0).numpy()
        Image.fromarray(np.ascontiguousarray(pixels)).save(
            output_dir / f"{frame_index + 1:06d}.png"
        )


@torch.inference_mode()
def evaluate(
    model: JointVQModel,
    tokenizer_name: str,
    dataset: str,
    dataset_root: Path,
    output_dir: Path,
    device: torch.device,
    max_sequences: int | None,
) -> dict:
    sequences = dataset_sequences(dataset, dataset_root)
    if max_sequences is not None:
        sequences = sequences[:max_sequences]
    source_frames = uvg_source_frames(tokenizer_name)
    lpips_vgg_metric = create_lpips_vgg(device)
    lpips_alex_metric = create_standard_lpips(device)
    dists_metric = DISTS().to(device).eval()
    reports = []

    for sequence_name, frames_dir in sequences:
        video, _ = load_png_sequence(frames_dir, source_frames, start_frame=1)
        prepared = prepare_uvg_input(video.float().div(255.0), tokenizer_name)

        torch.cuda.synchronize(device)
        started = time.perf_counter()
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
            bits = float(_estimated_bits(model.entropy_model, selected, indices).item())
            reconstruction = model.tokenizer.decode(
                model.tokenizer.post_quant(selected.to(projected.dtype))
            )
        torch.cuda.synchronize(device)
        total_ms = (time.perf_counter() - started) * 1000.0

        reconstruction = finish_uvg_reconstruction(reconstruction, prepared)
        reference = prepared.reference
        save_reconstruction(reconstruction, output_dir / "recon" / sequence_name)
        difference = reference - reconstruction
        squared_error = float(difference.square().sum().item())
        value_count = difference.numel()
        rate_pixels = prepared.rate_frames * reference.shape[-2] * reference.shape[-1]
        lpips_vgg = sequence_lpips(reference, reconstruction, lpips_vgg_metric, device=device, batch_size=1)
        lpips_alex = sequence_lpips(reference, reconstruction, lpips_alex_metric, device=device, batch_size=1)
        reports.append({
            "sequence": sequence_name,
            "frames": reference.shape[0],
            "rate_frames": prepared.rate_frames,
            "resolution": [reference.shape[-2], reference.shape[-1]],
            "psnr": -10.0 * math.log10(squared_error / value_count),
            "bpp": bits / rate_pixels,
            "lpips": lpips_vgg,
            "lpips_vgg": lpips_vgg,
            "lpips_alex": lpips_alex,
            "dists": _sequence_dists(reference, reconstruction, dists_metric, device),
            "latency_ms": total_ms / reference.shape[0],
            "bits": bits,
            "squared_error": squared_error,
            "value_count": value_count,
        })
        print(f"completed {sequence_name}", flush=True)

    total_frames = sum(report["frames"] for report in reports)
    total_rate_pixels = sum(
        report["rate_frames"] * report["resolution"][0] * report["resolution"][1]
        for report in reports
    )
    total_error = sum(report["squared_error"] for report in reports)
    total_values = sum(report["value_count"] for report in reports)
    return {
        "schema_version": 2,
        "dataset": DATASET_NAMES[dataset],
        "tokenizer": tokenizer_name,
        "protocol": {
            "sequences": [name for name, _ in sequences],
            "model_input_frames": TOKENIZER_SPECS[tokenizer_name].input_frames,
            "metric_frames": uvg_metric_frames(tokenizer_name),
            "bpp_denominator_frames": source_frames,
            "rate": "estimated categorical cross-entropy bits",
        },
        "aggregate": {
            "psnr": -10.0 * math.log10(total_error / total_values),
            "bpp": sum(report["bits"] for report in reports) / total_rate_pixels,
            "lpips": sum(report["lpips"] * report["frames"] for report in reports) / total_frames,
            "lpips_vgg": sum(report["lpips_vgg"] * report["frames"] for report in reports) / total_frames,
            "lpips_alex": sum(report["lpips_alex"] * report["frames"] for report in reports) / total_frames,
            "dists": sum(report["dists"] * report["frames"] for report in reports) / total_frames,
            "latency_ms": sum(report["latency_ms"] * report["frames"] for report in reports) / total_frames,
        },
        "sequences": reports,
    }


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
    evaluation = evaluate(
        model,
        loaded.name,
        args.dataset,
        args.dataset_root,
        args.output_dir,
        device,
        args.max_sequences,
    )
    evaluation["checkpoint_step"] = int(checkpoint["step"])
    write_json(args.output_dir / "metrics.json", evaluation)


if __name__ == "__main__":
    main()
