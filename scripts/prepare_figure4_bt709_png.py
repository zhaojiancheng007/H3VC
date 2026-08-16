#!/usr/bin/env python3
"""Prepare the first 96 RGB BT.709 frames for Figure 4 evaluation."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import re
import struct
import subprocess
import tempfile
from pathlib import Path


FRAMES = 96
HEIGHT = 1080
WIDTH = 1920
PIXEL_FORMAT = "yuv420p"
FFMPEG_FILTER = (
    "scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24"
)
DIMENSIONS = re.compile(r"_(?P<width>\d+)x(?P<height>\d+)_")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
EXPECTED_SEQUENCES = {
    "UVG": (
        "Beauty",
        "Bosphorus",
        "HoneyBee",
        "Jockey",
        "ReadySteadyGo",
        "ShakeNDry",
        "YachtRide",
    ),
    "MCL-JCV": tuple(f"videoSRC{index:02d}" for index in range(1, 31)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert UVG and MCL-JCV raw YUV to first-96 RGB BT.709 PNGs"
    )
    parser.add_argument("--uvg-root", type=Path, required=True)
    parser.add_argument("--mcl-jcv-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def _sequence_name(path: Path) -> str:
    marker = f"_{WIDTH}x{HEIGHT}_"
    if marker not in path.stem:
        raise ValueError(f"Cannot parse sequence name from {path}")
    return path.stem.split(marker, maxsplit=1)[0]


def _source_frame_count(path: Path) -> int:
    match = DIMENSIONS.search(path.name)
    if match is None:
        raise ValueError(f"Cannot parse dimensions from {path}")
    width = int(match.group("width"))
    height = int(match.group("height"))
    if (height, width) != (HEIGHT, WIDTH):
        raise ValueError(f"Expected {WIDTH}x{HEIGHT}, got {width}x{height}: {path}")
    frame_bytes = width * height * 3 // 2
    frames, remainder = divmod(path.stat().st_size, frame_bytes)
    if remainder:
        raise ValueError(f"Raw YUV byte count is not frame-aligned: {path}")
    if frames < FRAMES:
        raise ValueError(f"Expected at least {FRAMES} frames, got {frames}: {path}")
    return frames


def _discover(dataset: str, root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = sorted(root.glob("*.yuv"), key=_sequence_name)
    names = tuple(_sequence_name(path) for path in paths)
    if names != EXPECTED_SEQUENCES[dataset]:
        raise ValueError(
            f"{dataset} sequence set differs from the Figure 4 protocol: {names}"
        )
    return paths


def _verify_png(path: Path) -> None:
    with path.open("rb") as handle:
        header = handle.read(26)
    if len(header) != 26 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise ValueError(f"Invalid PNG header: {path}")
    width, height = struct.unpack(">II", header[16:24])
    bit_depth = header[24]
    color_type = header[25]
    if (height, width, bit_depth, color_type) != (HEIGHT, WIDTH, 8, 2):
        raise ValueError(
            f"Expected {WIDTH}x{HEIGHT} RGB8 PNG, got "
            f"{width}x{height}, bit_depth={bit_depth}, color_type={color_type}: {path}"
        )


def _convert(dataset: str, source: Path, output_root: Path) -> dict[str, object]:
    name = _sequence_name(source)
    source_frames = _source_frame_count(source)
    dataset_dir = output_root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_dir = dataset_dir / name
    if output_dir.exists():
        raise FileExistsError(output_dir)

    temporary = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=dataset_dir))
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            PIXEL_FORMAT,
            "-video_size",
            f"{WIDTH}x{HEIGHT}",
            "-i",
            str(source),
            "-frames:v",
            str(FRAMES),
            "-vf",
            FFMPEG_FILTER,
            "-start_number",
            "1",
            str(temporary / "%06d.png"),
        ],
        check=True,
    )
    pngs = sorted(temporary.glob("*.png"))
    expected_names = [f"{index:06d}.png" for index in range(1, FRAMES + 1)]
    if [path.name for path in pngs] != expected_names:
        raise ValueError(f"Expected contiguous first-{FRAMES} PNGs for {source}")
    for png in pngs:
        _verify_png(png)
    os.replace(temporary, output_dir)
    return {
        "dataset": dataset,
        "name": name,
        "source_yuv": str(source.resolve()),
        "source_frames": source_frames,
        "frames_dir": str(output_dir.resolve()),
        "frames": FRAMES,
    }


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(
            f"Refusing to write into non-empty output root: {args.output_root}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)

    sources = [
        *[("UVG", path) for path in _discover("UVG", args.uvg_root)],
        *[("MCL-JCV", path) for path in _discover("MCL-JCV", args.mcl_jcv_root)],
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(_convert, dataset, source, args.output_root)
            for dataset, source in sources
        ]
        sequences = [future.result() for future in concurrent.futures.as_completed(futures)]
    sequences.sort(key=lambda item: (str(item["dataset"]), str(item["name"])))

    manifest = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": {
            "frames_per_sequence": FRAMES,
            "resolution": [HEIGHT, WIDTH],
            "input_pixel_format": PIXEL_FORMAT,
            "input_color_matrix": "bt709",
            "input_range": "tv",
            "output_pixel_format": "rgb24_png",
            "output_range": "pc",
            "ffmpeg_filter": FFMPEG_FILTER,
        },
        "datasets": {
            dataset: {
                "sequence_count": len(EXPECTED_SEQUENCES[dataset]),
                "sequences": list(EXPECTED_SEQUENCES[dataset]),
            }
            for dataset in EXPECTED_SEQUENCES
        },
        "sequences": sequences,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}")
    print(f"sequences={len(sequences)} frames={len(sequences) * FRAMES}")


if __name__ == "__main__":
    main()
