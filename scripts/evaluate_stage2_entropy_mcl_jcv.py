from __future__ import annotations

import argparse
from dataclasses import asdict
import math
from pathlib import Path
import sys
import time

from DISTS_pytorch import DISTS
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.video_sequence import load_png_sequence
from entropy_model.model_categorical import CategoricalEntropyModel
from metrics.lpips import create_standard_lpips, sequence_lpips
from metrics.reporting import write_json
from scripts.stage2 import load_checkpoint
from tokenizer.stage3_vq import selected_code_vectors
from tokenizer.stage_backbones import TOKENIZER_SPECS
from tokenizer.stage_checkpoint import load_stage1_tokenizer
from training.config_stage2 import load_stage2_config
from training.evaluation_categorical import _estimated_bits, _sequence_dists
from training.uvg_protocol import (
    finish_uvg_reconstruction,
    prepare_uvg_input,
    uvg_metric_frames,
    uvg_source_frames,
)
from training.video_losses import create_lpips_vgg


MCL_JCV_SEQUENCES = tuple(f"videoSRC{index:02d}" for index in range(1, 31))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one unified Stage2 entropy checkpoint on MCL-JCV"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mcl-jcv-root",
        default="/luojix/data/MCL-JCV/figure4_bt709_first96",
    )
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


@torch.inference_mode()
def evaluate_mcl_jcv(
    tokenizer: torch.nn.Module,
    entropy_model: CategoricalEntropyModel,
    tokenizer_name: str,
    dataset_root: Path,
    lpips_vgg_metric: torch.nn.Module,
    lpips_alex_metric: torch.nn.Module,
    dists_metric: DISTS,
    device: torch.device,
) -> dict:
    tokenizer.eval()
    entropy_model.eval()
    source_frames = uvg_source_frames(tokenizer_name)
    reports = []

    for sequence_name in MCL_JCV_SEQUENCES:
        video, _ = load_png_sequence(
            dataset_root / sequence_name, source_frames, start_frame=1
        )
        prepared = prepare_uvg_input(video.float().div(255.0), tokenizer_name)

        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent = tokenizer.encode(prepared.model_input.to(device))
            projected = tokenizer.pre_quant(latent)
        flat = projected.permute(0, 2, 3, 4, 1).reshape(
            -1, projected.shape[1]
        ).float()
        flat_indices = tokenizer.quantizer._nearest_indices(flat)
        indices = flat_indices.reshape(
            projected.shape[0],
            projected.shape[2],
            projected.shape[3],
            projected.shape[4],
        )
        selected = selected_code_vectors(tokenizer.quantizer.embed, indices)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bits = float(_estimated_bits(entropy_model, selected, indices).item())
            reconstruction = tokenizer.decode(
                tokenizer.post_quant(selected.to(projected.dtype))
            )
        torch.cuda.synchronize(device)
        total_ms = (time.perf_counter() - started) * 1000.0

        reconstruction = finish_uvg_reconstruction(reconstruction, prepared)
        reference = prepared.reference
        difference = reference - reconstruction
        squared_error = float(difference.square().sum().item())
        value_count = difference.numel()
        rate_pixels = (
            prepared.rate_frames * reference.shape[-2] * reference.shape[-1]
        )
        reports.append(
            {
                "sequence": sequence_name,
                "frames": reference.shape[0],
                "rate_frames": prepared.rate_frames,
                "resolution": [reference.shape[-2], reference.shape[-1]],
                "psnr": -10.0 * math.log10(squared_error / value_count),
                "bpp": bits / rate_pixels,
                "lpips": sequence_lpips(
                    reference,
                    reconstruction,
                    lpips_vgg_metric,
                    device=device,
                    batch_size=1,
                ),
                "lpips_alex": sequence_lpips(
                    reference,
                    reconstruction,
                    lpips_alex_metric,
                    device=device,
                    batch_size=1,
                ),
                "dists": _sequence_dists(
                    reference, reconstruction, dists_metric, device
                ),
                "latency_ms": total_ms / reference.shape[0],
                "bits": bits,
                "squared_error": squared_error,
                "value_count": value_count,
            }
        )

    total_frames = sum(report["frames"] for report in reports)
    total_rate_pixels = sum(
        report["rate_frames"]
        * report["resolution"][0]
        * report["resolution"][1]
        for report in reports
    )
    total_error = sum(report["squared_error"] for report in reports)
    total_values = sum(report["value_count"] for report in reports)
    return {
        "schema_version": 2,
        "dataset": "MCL-JCV",
        "tokenizer": tokenizer_name,
        "protocol": {
            "sequences": list(MCL_JCV_SEQUENCES),
            "model_input_frames": TOKENIZER_SPECS[tokenizer_name].input_frames,
            "metric_frames": uvg_metric_frames(tokenizer_name),
            "bpp_denominator_frames": source_frames,
            "rate": "estimated categorical cross-entropy bits",
        },
        "aggregate": {
            "psnr": -10.0 * math.log10(total_error / total_values),
            "bpp": sum(report["bits"] for report in reports) / total_rate_pixels,
            "lpips": sum(
                report["lpips"] * report["frames"] for report in reports
            )
            / total_frames,
            "lpips_alex": sum(
                report["lpips_alex"] * report["frames"] for report in reports
            )
            / total_frames,
            "dists": sum(
                report["dists"] * report["frames"] for report in reports
            )
            / total_frames,
            "latency_ms": sum(
                report["latency_ms"] * report["frames"] for report in reports
            )
            / total_frames,
        },
        "sequences": reports,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    config = load_stage2_config(args.config)
    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint["config"] != asdict(config):
        raise ValueError("Stage2 checkpoint and evaluation config differ")

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
    entropy_model = CategoricalEntropyModel(config.model).to(device).eval()
    entropy_model.load_state_dict(checkpoint["model"], strict=True)
    lpips_metric = create_lpips_vgg(device)
    lpips_alex_metric = create_standard_lpips(device)
    dists_metric = DISTS().to(device).eval()

    evaluation = evaluate_mcl_jcv(
        tokenizer,
        entropy_model,
        loaded.name,
        Path(args.mcl_jcv_root),
        lpips_metric,
        lpips_alex_metric,
        dists_metric,
        device,
    )
    evaluation["checkpoint_step"] = int(checkpoint["step"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, evaluation)


if __name__ == "__main__":
    main()
