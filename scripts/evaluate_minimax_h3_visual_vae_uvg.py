from __future__ import annotations

import argparse
import hashlib
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
        description="Evaluate MiniMax-H3 VisualVAE reconstruction on 96-frame UVG"
    )
    parser.add_argument(
        "--model-dir",
        default=str(PROJECT_ROOT / "pretrained_ckpts/MiniMax-H3/FL2VA/video_vae"),
    )
    parser.add_argument("--uvg-root", default="/luojix/data/UVG")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
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


class H3VisualVAE:
    def __init__(self, model_dir: Path, device: torch.device) -> None:
        if not (model_dir / "minimax_h3_video_vae.py").is_file():
            raise FileNotFoundError(model_dir / "minimax_h3_video_vae.py")
        self.weights_path = model_dir / "source/model.safetensors"
        if not self.weights_path.is_file():
            raise FileNotFoundError(self.weights_path)

        sys.path.insert(0, str(model_dir.parent.resolve()))
        from video_vae.minimax_h3_video_vae import MiniMaxH3VideoVAE

        self.device = device
        self.vae = MiniMaxH3VideoVAE.from_pretrained(str(model_dir)).to(device).eval()
        if self.vae.vae_ratio != 16 or self.vae.vae_ratio_t != 4:
            raise RuntimeError(
                f"Unexpected H3 downsampling: t={self.vae.vae_ratio_t}, "
                f"spatial={self.vae.vae_ratio}"
            )

    def reconstruct(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        # Input is RGB [T,3,H,W] in [0,1]. The official processor applies
        # ImageNet normalization and crops the bottom/right edges to multiples of 16.
        model_input = reference.to(self.device).permute(1, 0, 2, 3).contiguous()
        frame_count = int(reference.shape[0])
        padded_frames = frame_count + self.vae.processor.align_video_length(
            frame_count, mode="pad", granularity="chunk"
        )
        temporal_padding = padded_frames - frame_count
        if temporal_padding:
            model_input = torch.cat(
                [model_input, model_input[:, -1:].expand(-1, temporal_padding, -1, -1)],
                dim=1,
            )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latents = self.vae.encode_videos(
                [model_input], transform_input=True, use_fp16_latent=True,
            )
            latent = latents[0]
            decoded = self.vae.decode_base(
                latent.unsqueeze(0), frame_num=padded_frames,
                process_image=False,
            )
        reconstruction = self.vae.processor.revert_tensor(decoded)
        reconstruction = reconstruction[:, :, :frame_count]
        reconstruction = reconstruction.squeeze(0).permute(1, 0, 2, 3).float().cpu()
        return reconstruction, latent, temporal_padding


def evaluate_sequence(
    video: torch.Tensor,
    *,
    tokenizer: H3VisualVAE,
    lpips_metric: torch.nn.Module,
    dists_metric: torch.nn.Module,
    device: torch.device,
    metric_batch_size: int,
) -> dict:
    source_height, source_width = map(int, video.shape[-2:])
    evaluated_height = source_height // 16 * 16
    evaluated_width = source_width // 16 * 16
    reference = (
        video[:, :, :evaluated_height, :evaluated_width].float().div(255.0)
    )

    (reconstruction, latent, temporal_padding), forward_ms = timed(
        device, lambda: tokenizer.reconstruct(reference)
    )
    if reconstruction.shape != reference.shape:
        raise RuntimeError(
            f"Reconstruction shape mismatch: {reconstruction.shape} vs {reference.shape}"
        )
    if not torch.isfinite(reconstruction).all():
        raise FloatingPointError("H3 reconstruction contains non-finite values")

    difference = reference - reconstruction
    squared_error = float(difference.square().sum().item())
    value_count = difference.numel()
    frame_count = int(reference.shape[0])
    raw_fp16_bits = int(latent.numel()) * 16
    pixels = frame_count * evaluated_height * evaluated_width

    return {
        "frames": frame_count,
        "source_resolution": [source_height, source_width],
        "evaluated_resolution": [evaluated_height, evaluated_width],
        "latent_shape_cthw": list(latent.shape),
        "latent_dtype": str(latent.dtype),
        "temporal_padding_frames": temporal_padding,
        "psnr": -10.0 * math.log10(squared_error / value_count),
        "bpp": None,
        "raw_fp16_latent_bpp": raw_fp16_bits / pixels,
        "raw_fp16_latent_bits": raw_fp16_bits,
        "lpips_vgg": sequence_lpips(
            reference, reconstruction, lpips_metric,
            device=device, batch_size=metric_batch_size,
        ),
        "dists": sequence_dists(
            reference, reconstruction, dists_metric,
            device=device, batch_size=metric_batch_size,
        ),
        "latency_ms": forward_ms / frame_count,
        "squared_error": squared_error,
        "value_count": value_count,
    }


def main() -> None:
    args = parse_args()
    if args.metric_batch_size <= 0:
        raise ValueError("--metric-batch-size must be positive")
    output_json = Path(args.output_json)
    if output_json.exists():
        raise FileExistsError(f"Refusing to overwrite {output_json}")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = H3VisualVAE(Path(args.model_dir), device)
    lpips_metric = lpips.LPIPS(
        net="vgg", version="0.1", lpips=True, spatial=False, verbose=False
    ).to(device).eval()
    dists_metric = DISTS().to(device).eval()

    sequence_names = UVG_SEQUENCE_NAMES[:1] if args.smoke_test else UVG_SEQUENCE_NAMES
    frame_count = 17 if args.smoke_test else UVG_FRAMES
    reports = []
    for name in sequence_names:
        frames_dir = (
            Path(args.uvg_root) / f"{name}_1920x1080_120fps_420_8bit_YUV_frames"
        )
        video, _ = load_png_sequence(frames_dir, frame_count, start_frame=1)
        report = evaluate_sequence(
            video,
            tokenizer=tokenizer,
            lpips_metric=lpips_metric,
            dists_metric=dists_metric,
            device=device,
            metric_batch_size=args.metric_batch_size,
        )
        report["sequence"] = name
        reports.append(report)
        print(
            f"{name}: PSNR={report['psnr']:.6f}, "
            f"LPIPS-VGG={report['lpips_vgg']:.8f}, "
            f"DISTS={report['dists']:.8f}, "
            f"raw-FP16-bpp={report['raw_fp16_latent_bpp']:.9f}, "
            f"latency={report['latency_ms']:.3f} ms/frame",
            flush=True,
        )

    total_frames = sum(item["frames"] for item in reports)
    total_pixels = sum(
        item["frames"] * item["evaluated_resolution"][0] * item["evaluated_resolution"][1]
        for item in reports
    )
    total_squared_error = sum(item["squared_error"] for item in reports)
    total_value_count = sum(item["value_count"] for item in reports)
    aggregate = {
        "frames": total_frames,
        "psnr": -10.0 * math.log10(total_squared_error / total_value_count),
        "bpp": None,
        "raw_fp16_latent_bpp": sum(item["raw_fp16_latent_bits"] for item in reports)
        / total_pixels,
        "lpips_vgg": sum(item["lpips_vgg"] * item["frames"] for item in reports)
        / total_frames,
        "dists": sum(item["dists"] * item["frames"] for item in reports)
        / total_frames,
        "latency_ms": sum(item["latency_ms"] * item["frames"] for item in reports)
        / total_frames,
    }
    result = {
        "schema_version": 1,
        "model": "MiniMax-H3 FL2VA VisualVAE f16t4d24",
        "smoke_test": args.smoke_test,
        "checkpoint": {
            "path": str(tokenizer.weights_path.resolve()),
            "sha256": sha256(tokenizer.weights_path),
        },
        "protocol": {
            "dataset": "UVG",
            "sequences": list(sequence_names),
            "frames_per_sequence": frame_count,
            "source_resolution": [1080, 1920],
            "evaluated_resolution": [1072, 1920],
            "spatial_crop": "official top-left crop to multiples of 16",
            "temporal_alignment": "repeat last frame to the next official chunk-aligned length; metrics use only the original frames",
            "posterior": f"official stochastic Gaussian sample; torch seed {args.seed}",
            "bpp": "not applicable: continuous latent has no quantizer or entropy bitstream",
            "raw_fp16_latent_bpp": "24-channel FP16 latent storage only; not compressed bpp",
            "lpips_vgg": "lpips 0.1 VGG, frame mean, input [-1,1]",
            "dists": "DISTS-pytorch, frame mean, input [0,1], no resize",
            "latency_ms": "VisualVAE encode plus decode milliseconds per source frame",
        },
        "aggregate": aggregate,
        "sequences": reports,
    }
    write_json(output_json, result)
    print(f"Wrote {output_json}", flush=True)


if __name__ == "__main__":
    main()
