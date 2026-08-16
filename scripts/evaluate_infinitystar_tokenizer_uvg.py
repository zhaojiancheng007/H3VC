from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import sys
import time

from DISTS_pytorch import DISTS
import lpips
import numpy as np
import torch
import torch.nn.functional as F


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
TOKENIZER_HEIGHT = 720
TOKENIZER_WIDTH = 1280


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the official InfinityStar tokenizer on 96-frame UVG"
    )
    parser.add_argument("--uvg-root", default="/luojix/data/UVG")
    parser.add_argument("--infinitystar-repo", default="/zjc/InfinityStar")
    parser.add_argument(
        "--checkpoint",
        default="/zjc/InfinityStar/checkpoints/infinitystar_videovae.pth",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--metric-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
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


def build_infinitystar(checkpoint: Path, repo: Path, device: torch.device):
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not (repo / "infinity" / "utils" / "load.py").is_file():
        raise FileNotFoundError(f"Official InfinityStar source not found at {repo}")
    sys.path.insert(0, str(repo.resolve()))

    from infinity.models.self_correction import SelfCorrection
    from infinity.schedules import get_encode_decode_func
    from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta
    from infinity.utils.arg_util import Args
    from infinity.utils.load import load_visual_tokenizer

    args = Args()
    args.pn = "0.90M"
    args.video_frames = 81
    args.vae_path = str(checkpoint)
    args.videovae = 10
    args.vae_type = 64
    args.dynamic_scale_schedule = "infinity_elegant_clip20frames_v2"
    args.image_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]"
    args.video_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1]"
    args.use_two_stage_lfq = 1
    args.detail_scale_min_tokens = 750
    args.semantic_scales = 12
    args.max_repeat_times = 10000
    args.device = str(device)

    vae = load_visual_tokenizer(args, device=device).float().eval()
    self_correction = SelfCorrection(vae, args)
    video_encode, video_decode, _, _ = get_encode_decode_func(
        args.dynamic_scale_schedule
    )
    dynamic_resolution_h_w, templates = get_dynamic_resolution_meta(
        args.dynamic_scale_schedule, args.video_frames
    )
    target_ratio = TOKENIZER_HEIGHT / TOKENIZER_WIDTH
    mapped_ratio = templates[np.argmin(np.abs(templates - target_ratio))]
    metadata = dynamic_resolution_h_w[mapped_ratio][args.pn]
    image_schedule = metadata["pt2scale_schedule"][1]
    if image_schedule[-1] != (1, TOKENIZER_HEIGHT // 16, TOKENIZER_WIDTH // 16):
        raise RuntimeError(
            f"Unexpected official 720p final scale: {image_schedule[-1]}"
        )
    return args, vae, self_correction, video_encode, video_decode, dynamic_resolution_h_w, mapped_ratio, image_schedule


def add_segment_schedule(metadata: dict, latent_frames: int, image_schedule: list) -> list:
    if latent_frames <= 1:
        raise ValueError(f"A video segment needs at least two latent frames, got {latent_frames}")
    schedule = list(image_schedule)
    schedule.extend(
        (latent_frames - 1, height, width)
        for _, height, width in image_schedule
    )
    metadata["pt2scale_schedule"][latent_frames] = schedule
    return schedule


def metric_pair(
    reference: torch.Tensor,
    reconstruction: torch.Tensor,
    *,
    lpips_metric: torch.nn.Module,
    dists_metric: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    return (
        sequence_lpips(
            reference,
            reconstruction,
            lpips_metric,
            device=device,
            batch_size=batch_size,
        ),
        sequence_dists(
            reference,
            reconstruction,
            dists_metric,
            device=device,
            batch_size=batch_size,
        ),
    )


def evaluate_sequence(
    video: torch.Tensor,
    *,
    segment_lengths: tuple[int, ...],
    args,
    vae,
    self_correction,
    video_encode,
    video_decode,
    dynamic_resolution_h_w: dict,
    mapped_ratio: float,
    image_schedule: list,
    lpips_metric: torch.nn.Module,
    dists_metric: torch.nn.Module,
    device: torch.device,
    metric_batch_size: int,
) -> dict:
    frame_offset = 0
    squared_error = 0.0
    value_count = 0
    lpips_sum = 0.0
    dists_sum = 0.0
    tokenizer_lpips_sum = 0.0
    tokenizer_dists_sum = 0.0
    raw_bits = 0
    encode_ms = 0.0
    decode_ms = 0.0
    segments = []
    metadata = dynamic_resolution_h_w[mapped_ratio][args.pn]

    for frame_count in segment_lengths:
        reference = video[frame_offset : frame_offset + frame_count].float().div(255.0)
        tokenizer_reference = F.interpolate(
            reference,
            size=(TOKENIZER_HEIGHT, TOKENIZER_WIDTH),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0.0, 1.0)
        model_input = (
            tokenizer_reference.permute(1, 0, 2, 3)
            .unsqueeze(0).mul(2.0).sub(1.0).to(device)
        )
        latent_frames = 1 + (frame_count - 1) // 4
        scale_schedule = add_segment_schedule(
            metadata, latent_frames, image_schedule
        )

        encoded, current_encode_ms = timed(
            device,
            lambda: video_encode(
                vae,
                model_input,
                vae_features=None,
                self_correction=self_correction,
                device=str(device),
                args=args,
                infer_mode=True,
                dynamic_resolution_h_w=dynamic_resolution_h_w,
            ),
        )
        noise_list, _, bit_indices, _, _, _ = encoded
        decoded, current_decode_ms = timed(
            device,
            lambda: video_decode(
                vae,
                bit_indices,
                scale_schedule,
                "bit_label",
                args=args,
                noise_list=noise_list,
            ),
        )
        tokenizer_reconstruction = (
            decoded.float().clamp(-1.0, 1.0).add(1.0).div(2.0)
            .squeeze(0).permute(1, 0, 2, 3).cpu()
        )
        if tokenizer_reconstruction.shape != tokenizer_reference.shape:
            raise RuntimeError(
                "InfinityStar reconstruction shape mismatch: "
                f"{tokenizer_reconstruction.shape} vs {tokenizer_reference.shape}"
            )
        reconstruction = F.interpolate(
            tokenizer_reconstruction,
            size=(reference.shape[2], reference.shape[3]),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0.0, 1.0)

        difference = reference - reconstruction
        current_squared_error = float(difference.square().sum().item())
        current_value_count = difference.numel()
        current_lpips, current_dists = metric_pair(
            reference,
            reconstruction,
            lpips_metric=lpips_metric,
            dists_metric=dists_metric,
            device=device,
            batch_size=metric_batch_size,
        )
        tokenizer_lpips, tokenizer_dists = metric_pair(
            tokenizer_reference,
            tokenizer_reconstruction,
            lpips_metric=lpips_metric,
            dists_metric=dists_metric,
            device=device,
            batch_size=metric_batch_size,
        )
        current_bits = sum(int(item.numel()) for item in bit_indices)

        squared_error += current_squared_error
        value_count += current_value_count
        lpips_sum += current_lpips * frame_count
        dists_sum += current_dists * frame_count
        tokenizer_lpips_sum += tokenizer_lpips * frame_count
        tokenizer_dists_sum += tokenizer_dists * frame_count
        raw_bits += current_bits
        encode_ms += current_encode_ms
        decode_ms += current_decode_ms
        segments.append(
            {
                "frames": frame_count,
                "latent_frames": latent_frames,
                "scale_schedule": [list(item) for item in scale_schedule],
                "bit_indices_shapes": [list(item.shape) for item in bit_indices],
                "raw_bits": current_bits,
                "psnr": -10.0 * math.log10(current_squared_error / current_value_count),
                "lpips_vgg": current_lpips,
                "dists": current_dists,
                "tokenizer_resolution_lpips_vgg": tokenizer_lpips,
                "tokenizer_resolution_dists": tokenizer_dists,
                "latency_ms": (current_encode_ms + current_decode_ms) / frame_count,
            }
        )
        frame_offset += frame_count
        del model_input, encoded, noise_list, bit_indices, decoded
        del tokenizer_reconstruction, reconstruction, reference, tokenizer_reference, difference

    source_pixels = frame_offset * int(video.shape[2]) * int(video.shape[3])
    return {
        "frames": frame_offset,
        "source_resolution": [int(video.shape[2]), int(video.shape[3])],
        "tokenizer_resolution": [TOKENIZER_HEIGHT, TOKENIZER_WIDTH],
        "psnr": -10.0 * math.log10(squared_error / value_count),
        "bpp": raw_bits / source_pixels,
        "raw_bits": raw_bits,
        "lpips_vgg": lpips_sum / frame_offset,
        "dists": dists_sum / frame_offset,
        "tokenizer_resolution_lpips_vgg": tokenizer_lpips_sum / frame_offset,
        "tokenizer_resolution_dists": tokenizer_dists_sum / frame_offset,
        "latency_ms": (encode_ms + decode_ms) / frame_offset,
        "squared_error": squared_error,
        "value_count": value_count,
        "segments": segments,
    }


def main() -> None:
    args_cli = parse_args()
    if args_cli.metric_batch_size <= 0:
        raise ValueError("--metric-batch-size must be positive")
    output_json = Path(args_cli.output_json)
    if output_json.exists():
        raise FileExistsError(f"Refusing to overwrite {output_json}")

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    device = torch.device(args_cli.device)
    checkpoint = Path(args_cli.checkpoint)
    built = build_infinitystar(
        checkpoint, Path(args_cli.infinitystar_repo), device
    )
    (
        args,
        vae,
        self_correction,
        video_encode,
        video_decode,
        dynamic_resolution_h_w,
        mapped_ratio,
        image_schedule,
    ) = built
    lpips_metric = lpips.LPIPS(
        net="vgg", version="0.1", lpips=True, spatial=False, verbose=False
    ).to(device).eval()
    dists_metric = DISTS().to(device).eval()

    sequence_names = UVG_SEQUENCE_NAMES[:1] if args_cli.smoke_test else UVG_SEQUENCE_NAMES
    segment_lengths = (25,) if args_cli.smoke_test else UVG_SEGMENT_LENGTHS
    sequence_reports = []
    for name in sequence_names:
        frames_dir = (
            Path(args_cli.uvg_root)
            / f"{name}_1920x1080_120fps_420_8bit_YUV_frames"
        )
        video, _ = load_png_sequence(frames_dir, sum(segment_lengths), start_frame=1)
        result = evaluate_sequence(
            video,
            segment_lengths=segment_lengths,
            args=args,
            vae=vae,
            self_correction=self_correction,
            video_encode=video_encode,
            video_decode=video_decode,
            dynamic_resolution_h_w=dynamic_resolution_h_w,
            mapped_ratio=mapped_ratio,
            image_schedule=image_schedule,
            lpips_metric=lpips_metric,
            dists_metric=dists_metric,
            device=device,
            metric_batch_size=args_cli.metric_batch_size,
        )
        result["sequence"] = name
        sequence_reports.append(result)
        print(
            f"{name}: bpp={result['bpp']:.9f}, PSNR={result['psnr']:.6f}, "
            f"LPIPS-VGG={result['lpips_vgg']:.8f}, DISTS={result['dists']:.8f}",
            flush=True,
        )

    total_frames = sum(item["frames"] for item in sequence_reports)
    total_source_pixels = sum(
        item["frames"] * item["source_resolution"][0] * item["source_resolution"][1]
        for item in sequence_reports
    )
    total_squared_error = sum(item["squared_error"] for item in sequence_reports)
    total_value_count = sum(item["value_count"] for item in sequence_reports)
    aggregate = {
        "frames": total_frames,
        "psnr": -10.0 * math.log10(total_squared_error / total_value_count),
        "bpp": sum(item["raw_bits"] for item in sequence_reports) / total_source_pixels,
        "lpips_vgg": sum(item["lpips_vgg"] * item["frames"] for item in sequence_reports) / total_frames,
        "dists": sum(item["dists"] * item["frames"] for item in sequence_reports) / total_frames,
        "tokenizer_resolution_lpips_vgg": sum(item["tokenizer_resolution_lpips_vgg"] * item["frames"] for item in sequence_reports) / total_frames,
        "tokenizer_resolution_dists": sum(item["tokenizer_resolution_dists"] * item["frames"] for item in sequence_reports) / total_frames,
        "latency_ms": sum(item["latency_ms"] * item["frames"] for item in sequence_reports) / total_frames,
    }
    report = {
        "schema_version": 1,
        "model": "InfinityStar official 64-bit multiscale BSQ tokenizer",
        "smoke_test": args_cli.smoke_test,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "protocol": {
            "dataset": "UVG",
            "frames_per_sequence": sum(segment_lengths),
            "segment_lengths": list(segment_lengths),
            "source_resolution": [1080, 1920],
            "tokenizer_resolution": [720, 1280],
            "primary_metrics": "720p tokenizer reconstruction bicubic-upsampled to source 1080p, compared with original UVG",
            "tokenizer_resolution_metrics": "720p reconstruction compared with bicubic-downsampled 720p UVG reference",
            "rate_mode": "raw direct multiscale BSQ bits from actual bit_indices, no entropy coding or container overhead",
            "bpp_denominator": "source frames times 1080 times 1920",
            "lpips_vgg": "lpips 0.1 VGG, frame arithmetic mean, input [-1,1]",
            "dists": "DISTS-pytorch, frame arithmetic mean, input [0,1]",
            "latency_ms": "tokenizer encode plus decode milliseconds per source frame",
            "schedule_extension": "official 720p spatial scales and repetitions; video scale temporal length is segment latent frames minus one",
            "seed": args_cli.seed,
        },
        "aggregate": aggregate,
        "sequences": sequence_reports,
    }
    write_json(output_json, report)
    print(f"Wrote {output_json}", flush=True)


if __name__ == "__main__":
    main()
