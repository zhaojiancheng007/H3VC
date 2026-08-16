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
from tokenizer.cosmos_dv.preprocessing import pad_for_cosmos, unpad_reconstruction


MODEL_ALIGNMENTS = {
    "Cosmos-0.1-Tokenizer-DV4x8x8": (4, 16),
    "Cosmos-0.1-Tokenizer-DV8x8x8": (8, 16),
    "Cosmos-0.1-Tokenizer-DV8x16x16": (8, 16),
}
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
        description="Evaluate an official Cosmos DV tokenizer on 96-frame UVG"
    )
    parser.add_argument("--model-id", choices=tuple(MODEL_ALIGNMENTS), required=True)
    parser.add_argument("--uvg-root", default="/luojix/data/UVG")
    parser.add_argument(
        "--checkpoint-root", default=str(PROJECT_ROOT / "pretrained_ckpts")
    )
    parser.add_argument(
        "--cosmos-repo", default=str(PROJECT_ROOT / "Cosmos-Tokenizer")
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metric-batch-size", type=int, default=1)
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


class CosmosDVVariant:
    def __init__(
        self,
        model_id: str,
        checkpoint_root: Path,
        cosmos_repo: Path,
        device: torch.device,
    ) -> None:
        model_dir = checkpoint_root / model_id
        self.encoder_path = model_dir / "encoder.jit"
        self.decoder_path = model_dir / "decoder.jit"
        if not self.encoder_path.is_file():
            raise FileNotFoundError(self.encoder_path)
        if not self.decoder_path.is_file():
            raise FileNotFoundError(self.decoder_path)
        if not (cosmos_repo / "cosmos_tokenizer" / "video_lib.py").is_file():
            raise FileNotFoundError(f"Official Cosmos source not found at {cosmos_repo}")

        sys.path.insert(0, str(cosmos_repo.resolve()))
        from cosmos_tokenizer.video_lib import CausalVideoTokenizer

        self.device = device
        self.encoder = CausalVideoTokenizer(
            checkpoint_enc=str(self.encoder_path),
            device=str(device),
            dtype="bfloat16",
        )
        self.decoder = CausalVideoTokenizer(
            checkpoint_dec=str(self.decoder_path),
            device=str(device),
            dtype="bfloat16",
        )

    def encode(self, model_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.jit.optimized_execution(False):
            output = self.encoder.encode(
                model_input.to(self.device, dtype=torch.bfloat16)
            )
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError(f"Unexpected Cosmos encoder output: {type(output)}")
        indices, codes = output
        if indices.ndim != 4 or codes.ndim != 5 or codes.shape[1] != 6:
            raise RuntimeError(
                f"Unexpected Cosmos shapes: indices={tuple(indices.shape)}, "
                f"codes={tuple(codes.shape)}"
            )
        return indices, codes

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        with torch.jit.optimized_execution(False):
            return self.decoder.decode(indices.to(self.device))


def evaluate_sequence(
    video: torch.Tensor,
    *,
    segment_lengths: tuple[int, ...],
    tokenizer: CosmosDVVariant,
    temporal_alignment: int,
    spatial_alignment: int,
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
    raw_bits = 0
    encode_ms = 0.0
    decode_ms = 0.0
    segments = []

    for frame_count in segment_lengths:
        reference = video[frame_offset : frame_offset + frame_count].float().div(255.0)
        model_input = reference.permute(1, 0, 2, 3).unsqueeze(0).mul(2.0).sub(1.0)
        model_input, pad_info = pad_for_cosmos(
            model_input,
            temporal_alignment=temporal_alignment,
            spatial_alignment=spatial_alignment,
        )

        (indices, codes), current_encode_ms = timed(
            device, lambda: tokenizer.encode(model_input)
        )
        decoded, current_decode_ms = timed(
            device, lambda: tokenizer.decode(indices)
        )
        decoded = unpad_reconstruction(decoded, pad_info)
        reconstruction = (
            decoded.float().clamp(-1.0, 1.0).add(1.0).div(2.0)
            .squeeze(0).permute(1, 0, 2, 3).cpu()
        )
        if reconstruction.shape != reference.shape:
            raise RuntimeError(
                f"Reconstruction shape mismatch: {reconstruction.shape} vs {reference.shape}"
            )

        difference = reference - reconstruction
        current_squared_error = float(difference.square().sum().item())
        current_value_count = difference.numel()
        current_lpips = sequence_lpips(
            reference,
            reconstruction,
            lpips_metric,
            device=device,
            batch_size=metric_batch_size,
        )
        current_dists = sequence_dists(
            reference,
            reconstruction,
            dists_metric,
            device=device,
            batch_size=metric_batch_size,
        )
        current_bits = int(indices.numel()) * 16

        squared_error += current_squared_error
        value_count += current_value_count
        lpips_sum += current_lpips * frame_count
        dists_sum += current_dists * frame_count
        raw_bits += current_bits
        encode_ms += current_encode_ms
        decode_ms += current_decode_ms
        segments.append(
            {
                "frames": frame_count,
                "padded_shape_bcthw": list(model_input.shape),
                "indices_shape": list(indices.shape),
                "codes_shape": list(codes.shape),
                "raw_bits": current_bits,
                "psnr": -10.0 * math.log10(current_squared_error / current_value_count),
                "lpips_vgg": current_lpips,
                "dists": current_dists,
                "latency_ms": (current_encode_ms + current_decode_ms) / frame_count,
            }
        )
        frame_offset += frame_count
        del model_input, indices, codes, decoded, reconstruction, reference, difference

    pixels = frame_offset * int(video.shape[2]) * int(video.shape[3])
    return {
        "frames": frame_offset,
        "resolution": [int(video.shape[2]), int(video.shape[3])],
        "psnr": -10.0 * math.log10(squared_error / value_count),
        "bpp": raw_bits / pixels,
        "raw_bits": raw_bits,
        "lpips_vgg": lpips_sum / frame_offset,
        "dists": dists_sum / frame_offset,
        "latency_ms": (encode_ms + decode_ms) / frame_offset,
        "squared_error": squared_error,
        "value_count": value_count,
        "segments": segments,
    }


def main() -> None:
    args = parse_args()
    if args.metric_batch_size <= 0:
        raise ValueError("--metric-batch-size must be positive")
    output_json = Path(args.output_json)
    if output_json.exists():
        raise FileExistsError(f"Refusing to overwrite {output_json}")

    device = torch.device(args.device)
    temporal_alignment, spatial_alignment = MODEL_ALIGNMENTS[args.model_id]
    tokenizer = CosmosDVVariant(
        args.model_id,
        Path(args.checkpoint_root),
        Path(args.cosmos_repo),
        device,
    )
    lpips_metric = lpips.LPIPS(
        net="vgg", version="0.1", lpips=True, spatial=False, verbose=False
    ).to(device).eval()
    dists_metric = DISTS().to(device).eval()

    sequence_names = UVG_SEQUENCE_NAMES[:1] if args.smoke_test else UVG_SEQUENCE_NAMES
    segment_lengths = (25,) if args.smoke_test else UVG_SEGMENT_LENGTHS
    sequence_reports = []
    for name in sequence_names:
        frames_dir = (
            Path(args.uvg_root)
            / f"{name}_1920x1080_120fps_420_8bit_YUV_frames"
        )
        video, _ = load_png_sequence(frames_dir, sum(segment_lengths), start_frame=1)
        result = evaluate_sequence(
            video,
            segment_lengths=segment_lengths,
            tokenizer=tokenizer,
            temporal_alignment=temporal_alignment,
            spatial_alignment=spatial_alignment,
            lpips_metric=lpips_metric,
            dists_metric=dists_metric,
            device=device,
            metric_batch_size=args.metric_batch_size,
        )
        result["sequence"] = name
        sequence_reports.append(result)
        print(
            f"{name}: bpp={result['bpp']:.9f}, PSNR={result['psnr']:.6f}, "
            f"LPIPS-VGG={result['lpips_vgg']:.8f}, DISTS={result['dists']:.8f}",
            flush=True,
        )

    total_frames = sum(item["frames"] for item in sequence_reports)
    total_pixels = sum(
        item["frames"] * item["resolution"][0] * item["resolution"][1]
        for item in sequence_reports
    )
    total_squared_error = sum(item["squared_error"] for item in sequence_reports)
    total_value_count = sum(item["value_count"] for item in sequence_reports)
    aggregate = {
        "frames": total_frames,
        "psnr": -10.0 * math.log10(total_squared_error / total_value_count),
        "bpp": sum(item["raw_bits"] for item in sequence_reports) / total_pixels,
        "lpips_vgg": sum(item["lpips_vgg"] * item["frames"] for item in sequence_reports) / total_frames,
        "dists": sum(item["dists"] * item["frames"] for item in sequence_reports) / total_frames,
        "latency_ms": sum(item["latency_ms"] * item["frames"] for item in sequence_reports) / total_frames,
    }
    report = {
        "schema_version": 1,
        "model": args.model_id,
        "smoke_test": args.smoke_test,
        "checkpoints": {
            "encoder": str(tokenizer.encoder_path.resolve()),
            "encoder_sha256": sha256(tokenizer.encoder_path),
            "decoder": str(tokenizer.decoder_path.resolve()),
            "decoder_sha256": sha256(tokenizer.decoder_path),
        },
        "protocol": {
            "dataset": "UVG",
            "frames_per_sequence": sum(segment_lengths),
            "segment_lengths": list(segment_lengths),
            "source_resolution": [1080, 1920],
            "input_range": "RGB [0,1] converted to Cosmos [-1,1]",
            "temporal_alignment": temporal_alignment,
            "spatial_alignment": spatial_alignment,
            "rate_mode": "raw direct FSQ indices, 16 bits/token, no entropy coding or container overhead",
            "lpips_vgg": "lpips 0.1 VGG, frame arithmetic mean, input [-1,1]",
            "dists": "DISTS-pytorch, frame arithmetic mean, input [0,1], no resize",
            "latency_ms": "tokenizer encode plus decode milliseconds per source frame",
        },
        "aggregate": aggregate,
        "sequences": sequence_reports,
    }
    write_json(output_json, report)
    print(f"Wrote {output_json}", flush=True)


if __name__ == "__main__":
    main()
