"""Reconstruct RGB frames and report bpp and LPIPS."""
import argparse
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.checkpoint import load_model
from inference.reconstruction import reconstruct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="H3VC inference checkpoint")
    parser.add_argument("--frames-dir", type=Path, required=True, help="RGB frame directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="output directory for reconstruction PNGs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda", 0)
    model = load_model(args.checkpoint, device)
    report = reconstruct(args.frames_dir, model.tokenizer, model.entropy_model,
                         device, args.output_dir / "recon")
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved reconstructions and metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
