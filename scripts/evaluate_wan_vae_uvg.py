from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from DISTS_pytorch import DISTS
import lpips
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.video_sequence import load_png_sequence
from metrics.lpips import sequence_lpips
from metrics.reporting import write_json


UVG_FRAMES = 96
UVG_SEGMENT_LENGTHS = (25, 25, 25, 21)
UVG_SEQUENCE_NAMES = (
    "Beauty",
    "Bosphorus",
    "HoneyBee",
    "Jockey",
    "ReadySteadyGo",
    "ShakeNDry",
    "YachtRide",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate official Wan2.1 VAE reconstruction on UVG"
    )
    parser.add_argument("--uvg-root", default="/luojix/data/UVG")
    parser.add_argument(
        "--vae-checkpoint",
        default=str(
            PROJECT_ROOT / "pretrained_ckpts" / "Wan2.1" / "Wan2.1_VAE.pth"
        ),
    )
    parser.add_argument("--wan-repo", default=str(PROJECT_ROOT / "Wan2.1"))
    parser.add_argument(
        "--figure4-json",
        default=str(
            PROJECT_ROOT
            / "plots"
            / "gnvc_vd_vs_glc_video"
            / "stage3_every10k"
            / "rd_points.json"
        ),
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metric-batch-size", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device: torch.device, function):
    synchronize(device)
    started = time.perf_counter()
    result = function()
    synchronize(device)
    return result, (time.perf_counter() - started) * 1000.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_figure4_thresholds(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    values: dict[str, list[float]] = {"LPIPS-VGG": [], "DISTS": []}
    for record in report["records"]:
        if record["dataset"] == "UVG" and record["method"] == "GNVC-VD":
            values[record["metric"]].append(float(record["value"]))
    if not all(values.values()):
        raise ValueError(f"Missing GNVC-VD UVG Figure 4 values in {path}")
    return {metric: min(metric_values) for metric, metric_values in values.items()}


@torch.inference_mode()
def sequence_dists(
    reference: torch.Tensor,
    reconstruction: torch.Tensor,
    metric: torch.nn.Module,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    if reference.shape != reconstruction.shape:
        raise ValueError(
            f"DISTS shape mismatch: {reference.shape} vs {reconstruction.shape}"
        )
    total = 0.0
    for start in range(0, reference.shape[0], batch_size):
        end = min(start + batch_size, reference.shape[0])
        values = metric(
            reference[start:end].to(device=device, dtype=torch.float32),
            reconstruction[start:end].to(device=device, dtype=torch.float32),
            require_grad=False,
            batch_average=False,
        )
        if values.numel() != end - start:
            raise RuntimeError(
                f"DISTS returned {values.numel()} values for {end - start} frames"
            )
        if not torch.isfinite(values).all():
            raise FloatingPointError("DISTS returned non-finite values")
        total += float(values.double().sum().item())
    return total / reference.shape[0]


def evaluate_clip(
    video: torch.Tensor,
    *,
    segment_lengths: tuple[int, ...],
    vae,
    lpips_metric: torch.nn.Module,
    dists_metric: torch.nn.Module,
    device: torch.device,
    metric_batch_size: int,
) -> dict:
    frame_offset = 0
    squared_error = 0.0
    value_count = 0
    lpips_frame_sum = 0.0
    dists_frame_sum = 0.0
    encode_ms = 0.0
    decode_ms = 0.0
    segments = []

    for segment_length in segment_lengths:
        frame_end = frame_offset + segment_length
        reference = video[frame_offset:frame_end].float().div(255.0)
        model_input = reference.permute(1, 0, 2, 3).mul(2.0).sub(1.0).to(device)

        latents, current_encode_ms = timed(device, lambda: vae.encode([model_input])[0])
        decoded, current_decode_ms = timed(device, lambda: vae.decode([latents])[0])
        reconstruction = decoded.add(1.0).div(2.0).permute(1, 0, 2, 3).cpu()
        if reconstruction.shape != reference.shape:
            raise RuntimeError(
                f"WanVAE shape mismatch: {reconstruction.shape} vs {reference.shape}"
            )

        difference = reference - reconstruction
        segment_squared_error = float(difference.square().sum().item())
        segment_value_count = difference.numel()
        segment_lpips = sequence_lpips(
            reference,
            reconstruction,
            lpips_metric,
            device=device,
            batch_size=metric_batch_size,
        )
        segment_dists = sequence_dists(
            reference,
            reconstruction,
            dists_metric,
            device=device,
            batch_size=metric_batch_size,
        )

        squared_error += segment_squared_error
        value_count += segment_value_count
        lpips_frame_sum += segment_lpips * segment_length
        dists_frame_sum += segment_dists * segment_length
        encode_ms += current_encode_ms
        decode_ms += current_decode_ms
        segments.append(
            {
                "frames": segment_length,
                "latent_shape": list(latents.shape),
                "psnr": -10.0
                * math.log10(segment_squared_error / segment_value_count),
                "lpips_vgg": segment_lpips,
                "dists": segment_dists,
                "latency_ms": {
                    "encode_total": current_encode_ms,
                    "decode_total": current_decode_ms,
                    "codec_per_frame": (current_encode_ms + current_decode_ms)
                    / segment_length,
                },
            }
        )
        frame_offset = frame_end
        del model_input, latents, decoded, reconstruction, reference, difference

    if frame_offset != video.shape[0]:
        raise RuntimeError(
            f"Segments consumed {frame_offset} of {video.shape[0]} input frames"
        )
    return {
        "frames": frame_offset,
        "resolution": [int(video.shape[2]), int(video.shape[3])],
        "psnr": -10.0 * math.log10(squared_error / value_count),
        "bpp": None,
        "lpips_vgg": lpips_frame_sum / frame_offset,
        "dists": dists_frame_sum / frame_offset,
        "latency_ms": (encode_ms + decode_ms) / frame_offset,
        "latency_breakdown_ms": {
            "encode_total": encode_ms,
            "decode_total": decode_ms,
            "codec_total": encode_ms + decode_ms,
        },
        "squared_error": squared_error,
        "value_count": value_count,
        "segments": segments,
    }


def main() -> None:
    args = parse_args()
    if args.metric_batch_size <= 0:
        raise ValueError("--metric-batch-size must be positive")
    if sum(UVG_SEGMENT_LENGTHS) != UVG_FRAMES:
        raise RuntimeError("UVG segment lengths must sum to 96")

    output_json = Path(args.output_json)
    if output_json.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {output_json}")
    checkpoint = Path(args.vae_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    wan_repo = Path(args.wan_repo)
    if not (wan_repo / "wan" / "modules" / "vae.py").is_file():
        raise FileNotFoundError(f"Official WanVAE source not found under {wan_repo}")
    sys.path.insert(0, str(wan_repo))
    from wan.modules.vae import WanVAE

    device = torch.device(args.device)
    vae = WanVAE(
        z_dim=16,
        vae_pth=str(checkpoint),
        dtype=torch.float32,
        device=device,
    )
    lpips_metric = lpips.LPIPS(
        net="vgg",
        version="0.1",
        lpips=True,
        spatial=False,
        verbose=False,
    ).to(device).eval()
    dists_metric = DISTS().to(device).eval()

    if args.dry_run:
        torch.manual_seed(0)
        sequence_names = ("synthetic",)
        segment_lengths = (5,)
    else:
        sequence_names = UVG_SEQUENCE_NAMES
        segment_lengths = UVG_SEGMENT_LENGTHS

    sequence_reports = []
    for name in sequence_names:
        if args.dry_run:
            video = torch.randint(0, 256, (5, 3, 64, 64), dtype=torch.uint8)
        else:
            frames_dir = (
                Path(args.uvg_root)
                / f"{name}_1920x1080_120fps_420_8bit_YUV_frames"
            )
            video, _ = load_png_sequence(frames_dir, UVG_FRAMES, start_frame=1)
        result = evaluate_clip(
            video,
            segment_lengths=segment_lengths,
            vae=vae,
            lpips_metric=lpips_metric,
            dists_metric=dists_metric,
            device=device,
            metric_batch_size=args.metric_batch_size,
        )
        result["sequence"] = name
        sequence_reports.append(result)
        print(
            f"{name}: PSNR={result['psnr']:.6f}, "
            f"LPIPS-VGG={result['lpips_vgg']:.8f}, "
            f"DISTS={result['dists']:.8f}",
            flush=True,
        )

    total_frames = sum(result["frames"] for result in sequence_reports)
    total_squared_error = sum(result["squared_error"] for result in sequence_reports)
    total_value_count = sum(result["value_count"] for result in sequence_reports)
    aggregate = {
        "frames": total_frames,
        "psnr": -10.0 * math.log10(total_squared_error / total_value_count),
        "bpp": None,
        "lpips_vgg": sum(
            result["lpips_vgg"] * result["frames"] for result in sequence_reports
        )
        / total_frames,
        "dists": sum(
            result["dists"] * result["frames"] for result in sequence_reports
        )
        / total_frames,
        "latency_ms": sum(
            result["latency_ms"] * result["frames"] for result in sequence_reports
        )
        / total_frames,
    }

    if args.dry_run:
        figure4 = None
        validation = None
    else:
        thresholds = load_figure4_thresholds(Path(args.figure4_json))
        figure4 = {
            "source": str(Path(args.figure4_json).resolve()),
            "gnvc_vd_best_lpips_vgg": thresholds["LPIPS-VGG"],
            "gnvc_vd_best_dists": thresholds["DISTS"],
        }
        validation = {
            "criterion": "Both WanVAE metrics must be lower than the best GNVC-VD UVG Figure 4 values",
            "lpips_vgg_pass": aggregate["lpips_vgg"] < thresholds["LPIPS-VGG"],
            "dists_pass": aggregate["dists"] < thresholds["DISTS"],
        }
        validation["passed"] = (
            validation["lpips_vgg_pass"] and validation["dists_pass"]
        )

    report = {
        "schema_version": 1,
        "model": "official_wan2.1_vae",
        "vae_checkpoint": str(checkpoint.resolve()),
        "vae_checkpoint_sha256": sha256(checkpoint),
        "dry_run": args.dry_run,
        "protocol": {
            "dataset": "synthetic" if args.dry_run else "UVG",
            "frames_per_sequence": 5 if args.dry_run else UVG_FRAMES,
            "segment_lengths": list(segment_lengths),
            "input_range": "RGB [0,1] converted to WanVAE [-1,1]",
            "latent": "deterministic normalized encoder mean, 16 channels",
            "dtype": "float32",
            "lpips_vgg": "lpips 0.1, VGG, frame arithmetic mean, input [-1,1]",
            "dists": "DISTS-pytorch 0.1, frame arithmetic mean, input [0,1], no resize",
            "rate_mode": "reconstruction_upper_bound_no_bitstream",
            "latency_ms": "WanVAE encode plus decode milliseconds per frame",
        },
        "aggregate": aggregate,
        "sequences": sequence_reports,
        "gnvc_vd_figure4": figure4,
        "validation": validation,
    }
    write_json(output_json, report)
    print(f"Wrote {output_json}", flush=True)


if __name__ == "__main__":
    main()
