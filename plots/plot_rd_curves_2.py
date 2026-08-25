#!/usr/bin/env python3
"""Plot the second RD figure with explicitly separated ARCodec curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import plot_rd_curves as rd
from plot_rd_curves import METRICS, baseline_records, manifest_records, read_json


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT.parent.parent / "eval_archive" / "manifest_rd_curve_2.json"
CURVE2_OURS_STYLES = (
    rd.OURS_STYLES[0],
    {"color": "#008C95", "marker": "*"},
    *rd.OURS_STYLES[2:],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, default=ROOT / "baseline_rd_points.json"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    return parser.parse_args()


def apply_curve_names(payload: dict, ours: list[dict]) -> list[dict]:
    curve_by_point = {
        (
            point["dataset"].removesuffix("_96frames"),
            float(point["aggregate"]["bpp"]),
        ): point["curve"]
        for point in payload["points"]
        if point["model"] == "ARCodec" and "curve" in point
    }
    for record in ours:
        curve = curve_by_point.get((record["dataset"], record["bpp"]))
        if curve is not None:
            record["method"] = curve
    return ours


def main() -> None:
    args = parse_args()
    payload = read_json(args.manifest)
    datasets, archived_baseline, ours = manifest_records(payload)
    baseline = baseline_records(read_json(args.baseline)) + archived_baseline
    rd.OURS_STYLES = CURVE2_OURS_STYLES
    rd.plot(
        datasets,
        baseline,
        apply_curve_names(payload, ours),
        args.output_dir,
        metrics=METRICS,
    )
    print(args.output_dir / "rd_curves.png")
    print(args.output_dir / "rd_curves.pdf")


if __name__ == "__main__":
    main()
