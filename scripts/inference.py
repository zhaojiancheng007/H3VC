"""Reconstruct the first 96 RGB PNG frames and report estimated bpp."""
import argparse
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.checkpoint import load_model
from inference.reconstruction import reconstruct
from tokenizer.gaussian_tile_decode import gaussian_tile_blend_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Trusted Stage3 checkpoint")
    parser.add_argument("--frames-dir", type=Path, required=True, help="RGB PNGs numbered 000001.png through 000096.png")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda", 0)
    model, step, architecture = load_model(args.checkpoint, device)
    report = reconstruct(args.frames_dir, model.tokenizer, model.entropy_model,
                         device, args.output_dir / "recon")
    report["checkpoint_step"] = step
    report["estimated_bits_per_token"] = report["bits"] / report["codebook"]["total_assignments"]
    report["protocol"] = {
        "rate": "estimated categorical cross-entropy; not an encoded bitstream",
        "precision": "BF16 autocast; FP32 quantization and cross-entropy",
        "metric_frames": 96, "model_input_frames": 107, "bpp_denominator_frames": 96,
        "entropy_k": 1, "entropy_architecture": architecture,
        "tile_blend": gaussian_tile_blend_protocol(),
        "latency_scope": "encode + VQ + entropy estimation + decode; excludes model loading, metrics and PNG IO",
        "latency_unit": "ms per source frame; single run, not a benchmark",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved reconstructions and metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
