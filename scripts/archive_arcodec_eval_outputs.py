from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


REQUIRED_METRICS = (
    "bpp",
    "psnr",
    "lpips_vgg",
    "lpips_alex",
    "dists",
    "latency_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--lambda-value", required=True, type=float)
    parser.add_argument("--expected-sequences", required=True, type=int)
    parser.add_argument("--expected-frames", required=True, type=int)
    return parser.parse_args()


def link_tree(source: Path, destination: Path) -> int:
    files = sorted(path for path in source.rglob("*") if path.is_file())
    for source_file in files:
        destination_file = destination / source_file.relative_to(source)
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        if destination_file.exists():
            source_stat = source_file.stat()
            destination_stat = destination_file.stat()
            if (source_stat.st_dev, source_stat.st_ino) != (
                destination_stat.st_dev,
                destination_stat.st_ino,
            ):
                raise FileExistsError(destination_file)
            continue
        os.link(source_file, destination_file)
    return len(files)


def main() -> None:
    args = parse_args()
    metrics_path = args.source / "metrics.json"
    recon_path = args.source / "recon"
    metrics = json.loads(metrics_path.read_text())
    sequences = metrics["sequences"]

    if len(sequences) != args.expected_sequences:
        raise ValueError(
            f"{args.source}: expected {args.expected_sequences} sequences, "
            f"found {len(sequences)}"
        )
    for scope in (metrics["aggregate"], *sequences):
        missing = [name for name in REQUIRED_METRICS if name not in scope]
        if missing:
            raise KeyError(f"{args.source}: missing metrics {missing}")
    for sequence in sequences:
        frames = list((recon_path / sequence["sequence"]).glob("*.png"))
        if len(frames) != args.expected_frames:
            raise ValueError(
                f"{args.source}: {sequence['sequence']} expected "
                f"{args.expected_frames} recon frames, found {len(frames)}"
            )

    dataset = {
        "UVG": "UVG_96frames",
        "MCL-JCV": "MCL-JCV_96frames",
    }[metrics["dataset"]]
    archived_metrics = {
        "schema_version": 1,
        "model": "ARCodec",
        "dataset": dataset,
        "rate_point": {
            "lambda": args.lambda_value,
            "checkpoint_step": metrics["checkpoint_step"],
            "bpp": metrics["aggregate"]["bpp"],
        },
        "protocol": metrics["protocol"],
        "aggregate": {
            name: metrics["aggregate"][name] for name in REQUIRED_METRICS
        },
        "sequences": [
            {
                "sequence": sequence["sequence"],
                **{name: sequence[name] for name in REQUIRED_METRICS},
            }
            for sequence in sequences
        ],
    }

    args.destination.mkdir(parents=True, exist_ok=True)
    recon_files = link_tree(recon_path, args.destination / "recon")
    (args.destination / "metrics.json").write_text(
        json.dumps(archived_metrics, indent=2, sort_keys=True) + "\n"
    )
    shutil.copy2(metrics_path, args.destination / "metrics.full.json")
    print(
        json.dumps(
            {
                "source": str(args.source),
                "destination": str(args.destination),
                "sequences": len(sequences),
                "recon_files": recon_files,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
