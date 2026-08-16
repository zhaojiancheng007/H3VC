from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

from DISTS_pytorch import DISTS
import lpips
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.video_sequence import load_png_sequence
from metrics.lpips import sequence_lpips
from metrics.reporting import write_json


UVG_FRAMES = 96
UVG_SEQUENCES = (
    "Beauty",
    "Bosphorus",
    "HoneyBee",
    "Jockey",
    "ReadySteadyGo",
    "ShakeNDry",
    "YachtRide",
)
TEMPORAL_MODEL_FRAMES = 107


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MiniMax-H3 temporal chunking against one direct 96-frame forward"
    )
    parser.add_argument(
        "--model-dir",
        default=str(PROJECT_ROOT / "pretrained_ckpts/MiniMax-H3/FL2VA/video_vae"),
    )
    parser.add_argument("--uvg-root", default="/luojix/data/UVG")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequence-limit", type=int, default=len(UVG_SEQUENCES))
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


@torch.inference_mode()
def sequence_dists(
    reference: torch.Tensor,
    reconstruction: torch.Tensor,
    metric: torch.nn.Module,
    device: torch.device,
) -> float:
    total = 0.0
    for frame in range(reference.shape[0]):
        value = metric(
            reference[frame : frame + 1].to(device),
            reconstruction[frame : frame + 1].to(device),
            require_grad=False,
            batch_average=False,
        )
        total += float(value.item())
    return total / reference.shape[0]


class H3TemporalAblation:
    def __init__(self, model_dir: Path, device: torch.device) -> None:
        sys.path.insert(0, str(model_dir.parent.resolve()))
        from video_vae.minimax_h3_video_vae import MiniMaxH3VideoVAE

        self.device = device
        self.vae = MiniMaxH3VideoVAE.from_pretrained(str(model_dir)).to(device).eval()
        if self.vae.vae_ratio_t != 4 or self.vae.vae_ratio != 16:
            raise RuntimeError(
                f"Unexpected H3 ratios: temporal={self.vae.vae_ratio_t}, spatial={self.vae.vae_ratio}"
            )

    def _normalize(self, video: torch.Tensor) -> torch.Tensor:
        return self.vae.processor.transform_tensor(video)

    def reconstruct(
        self, video: torch.Tensor, mode: str
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        torch.cuda.reset_peak_memory_stats(self.device)
        synchronize(self.device)
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if mode in ("temporal_107", "direct_107"):
                tail = video[:, :, -1:].expand(
                    -1, -1, TEMPORAL_MODEL_FRAMES - UVG_FRAMES, -1, -1
                )
                video = torch.cat((video, tail), dim=2)
            normalized = self._normalize(video)
            if mode == "temporal_107":
                moments = self.vae.encode_temporal(normalized)
                posterior_mean, _ = moments.chunk(2, dim=1)
                decoded = self.vae.decode_base(
                    posterior_mean,
                    frame_num=TEMPORAL_MODEL_FRAMES,
                    process_image=False,
                )
            elif mode in ("direct_107", "direct_96"):
                moments = self.vae._adaptive_encode(normalized)
                posterior_mean, _ = moments.chunk(2, dim=1)
                decoded = self.vae._adaptive_decode(posterior_mean)
                if mode == "direct_107":
                    decoded = self.vae.trim_output(decoded, TEMPORAL_MODEL_FRAMES)
            else:
                raise ValueError(mode)
            decoded = decoded[:, :, :UVG_FRAMES]
        synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        peak_memory_gib = torch.cuda.max_memory_allocated(self.device) / 1024**3
        if decoded.shape[2] != UVG_FRAMES:
            raise RuntimeError(f"{mode} decoded {decoded.shape[2]} frames, expected 96")
        reconstruction = self.vae.processor.revert_tensor(decoded)
        return reconstruction, posterior_mean, elapsed_ms, peak_memory_gib


def evaluate_mode(
    *,
    name: str,
    reference: torch.Tensor,
    reconstruction: torch.Tensor,
    latent: torch.Tensor,
    elapsed_ms: float,
    peak_memory_gib: float,
    lpips_metric: torch.nn.Module,
    dists_metric: torch.nn.Module,
    device: torch.device,
) -> dict:
    reconstruction = (
        reconstruction[:, :, :, : reference.shape[-2], : reference.shape[-1]]
        .float()
        .clamp(0.0, 1.0)
        .squeeze(0)
        .permute(1, 0, 2, 3)
        .cpu()
    )
    difference = reference - reconstruction
    squared_error = float(difference.square().sum().item())
    value_count = difference.numel()
    pixels = reference.shape[0] * reference.shape[-2] * reference.shape[-1]
    raw_fp16_bits = latent.numel() * 16
    return {
        "sequence": name,
        "frames": int(reference.shape[0]),
        "latent_shape_bcthw": list(latent.shape),
        "psnr": -10.0 * math.log10(squared_error / value_count),
        "bpp": None,
        "raw_fp16_latent_bpp": raw_fp16_bits / pixels,
        "raw_fp16_latent_bits": raw_fp16_bits,
        "lpips": sequence_lpips(
            reference, reconstruction, lpips_metric, device=device, batch_size=1
        ),
        "dists": sequence_dists(reference, reconstruction, dists_metric, device),
        "latency_ms": elapsed_ms / reference.shape[0],
        "peak_memory_gib": peak_memory_gib,
        "squared_error": squared_error,
        "value_count": value_count,
    }


def aggregate(reports: list[dict]) -> dict:
    frames = sum(report["frames"] for report in reports)
    return {
        "psnr": -10.0
        * math.log10(
            sum(report["squared_error"] for report in reports)
            / sum(report["value_count"] for report in reports)
        ),
        "bpp": None,
        "raw_fp16_latent_bpp": sum(
            report["raw_fp16_latent_bits"] for report in reports
        )
        / (frames * 1080 * 1920),
        "lpips": sum(report["lpips"] * report["frames"] for report in reports) / frames,
        "dists": sum(report["dists"] * report["frames"] for report in reports) / frames,
        "latency_ms": sum(
            report["latency_ms"] * report["frames"] for report in reports
        )
        / frames,
        "peak_memory_gib": max(report["peak_memory_gib"] for report in reports),
    }


def main() -> None:
    args = parse_args()
    if not 1 <= args.sequence_limit <= len(UVG_SEQUENCES):
        raise ValueError(f"sequence-limit must be in [1, {len(UVG_SEQUENCES)}]")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model = H3TemporalAblation(Path(args.model_dir), device)
    lpips_metric = lpips.LPIPS(
        net="vgg", version="0.1", lpips=True, spatial=False, verbose=False
    ).to(device).eval()
    dists_metric = DISTS().to(device).eval()
    reports = {"temporal_107": [], "direct_107": [], "direct_96": []}

    for name in UVG_SEQUENCES[: args.sequence_limit]:
        frames_dir = (
            Path(args.uvg_root) / f"{name}_1920x1080_120fps_420_8bit_YUV_frames"
        )
        source, _ = load_png_sequence(frames_dir, UVG_FRAMES, start_frame=1)
        reference = source.float().div(255.0)
        padded = F.pad(reference, (0, 0, 0, 8), mode="replicate")
        video = padded.permute(1, 0, 2, 3).unsqueeze(0).to(device)

        for mode in reports:
            reconstruction, latent, elapsed_ms, peak_memory_gib = model.reconstruct(
                video, mode
            )
            report = evaluate_mode(
                name=name,
                reference=reference,
                reconstruction=reconstruction,
                latent=latent,
                elapsed_ms=elapsed_ms,
                peak_memory_gib=peak_memory_gib,
                lpips_metric=lpips_metric,
                dists_metric=dists_metric,
                device=device,
            )
            reports[mode].append(report)
            print(
                f"{mode} {name}: PSNR={report['psnr']:.4f} "
                f"LPIPS={report['lpips']:.6f} DISTS={report['dists']:.6f} "
                f"latency={report['latency_ms']:.2f}ms/frame "
                f"peak={report['peak_memory_gib']:.2f}GiB",
                flush=True,
            )
            del reconstruction, latent
            torch.cuda.empty_cache()

    result = {
        "schema_version": 1,
        "model": "MiniMax-H3 FL2VA continuous VisualVAE",
        "protocol": {
            "dataset": "UVG",
            "sequences": list(UVG_SEQUENCES[: args.sequence_limit]),
            "frames": 96,
            "model_resolution": [1088, 1920],
            "metric_resolution": [1080, 1920],
            "spatial_padding": "replicate bottom 8 pixels",
            "posterior": "mean",
            "temporal_107": "96 repeat-padded to 107, encode_temporal/decode_temporal, metric crop to 96",
            "direct_107": "same 107-frame input, one _adaptive_encode/_adaptive_decode forward, metric crop to 96",
            "direct_96": "one 96-frame _adaptive_encode/_adaptive_decode forward",
            "bpp": "not applicable: continuous latent has no quantized bitstream",
        },
        "modes": {
            mode: {"aggregate": aggregate(mode_reports), "sequences": mode_reports}
            for mode, mode_reports in reports.items()
        },
    }
    write_json(output, result)
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
