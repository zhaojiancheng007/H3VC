from __future__ import annotations

import argparse
import math
from pathlib import Path

from DISTS_pytorch import DISTS
import lpips
import torch

from evaluate_cosmos_cv4x8x8_uvg import (
    CosmosCVVariant,
    PROJECT_ROOT,
    UVG_SEGMENT_LENGTHS,
    UVG_SEQUENCE_NAMES,
    evaluate_sequence,
    load_png_sequence,
    sha256,
    write_json,
)


MODEL_ALIGNMENTS = {
    "Cosmos-0.1-Tokenizer-CV8x8x8": (8, 16),
    "Cosmos-0.1-Tokenizer-CV8x16x16": (8, 16),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate official Cosmos CV8 tokenizer variants on 96-frame UVG"
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


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json)
    if output_json.exists():
        raise FileExistsError(f"Refusing to overwrite {output_json}")

    device = torch.device(args.device)
    temporal_alignment, spatial_alignment = MODEL_ALIGNMENTS[args.model_id]
    tokenizer = CosmosCVVariant(
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
        "lpips_vgg": sum(item["lpips_vgg"] * item["frames"] for item in sequence_reports)
        / total_frames,
        "dists": sum(item["dists"] * item["frames"] for item in sequence_reports)
        / total_frames,
        "latency_ms": sum(item["latency_ms"] * item["frames"] for item in sequence_reports)
        / total_frames,
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
            "rate_mode": "raw BF16 continuous latent, 16 bits/element, no quantization, entropy coding, or container overhead",
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
