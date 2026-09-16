"""Reconstruct one 96-frame PNG sequence and report estimated rate."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate import load_model
from DISTS_pytorch import DISTS
from metrics.lpips import create_standard_lpips
from metrics.reporting import write_json
from training.video_losses import create_lpips_vgg
from training.evaluation_categorical_distributed_codebook import (
    SequenceJob, evaluate_sequence_job,
)
from tokenizer.gaussian_tile_decode import gaussian_tile_blend_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tokenizer-checkpoint', required=True)
    parser.add_argument('--entropy-checkpoint', required=True)
    parser.add_argument('--minimax-h3-model-dir', required=True)
    parser.add_argument('--frames-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model, loaded, step, architecture, device = load_model(args)
    report = evaluate_sequence_job(
        SequenceJob('input', args.frames_dir.name, args.frames_dir),
        model.tokenizer, model.entropy_model, loaded.name,
        create_lpips_vgg(device), create_standard_lpips(device),
        DISTS().to(device).eval(), device, args.output_dir / 'recon', k=1,
    )
    report['checkpoint_step'] = step
    report['estimated_bits_per_token'] = report['bits'] / report['codebook']['total_assignments']
    report['protocol'] = {
        'rate': 'estimated categorical cross-entropy bits; not an encoded bitstream',
        'metric_frames': 96, 'model_input_frames': 107, 'bpp_denominator_frames': 96,
        'entropy_k': 1, 'entropy_architecture': architecture,
        'tile_blend': gaussian_tile_blend_protocol(),
    }
    write_json(args.output_dir / 'metrics.json', report)
    print(args.output_dir / 'metrics.json')


if __name__ == '__main__':
    main()
