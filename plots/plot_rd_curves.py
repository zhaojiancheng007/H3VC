#!/usr/bin/env python3
"""Plot GNVC-VD, GLC-Video, GVC1D, and ARCodec perceptual RD curves."""

from __future__ import annotations

import argparse
import json
from itertools import cycle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT.parent.parent / "eval_archive" / "manifest.json"
METRICS = (("LPIPS-VGG", "lpips_vgg"), ("DISTS", "dists"))
LEGACY_METRICS = METRICS
BASELINE_STYLES = {
    "GLC-Video": {
        "color": "#1f77b4",
        "marker": "v",
        "linestyle": "--",
    },
    "GNVC-VD": {
        "color": "#d62728",
        "marker": "D",
        "linestyle": "-",
    },
    "GVC1D": {
        "color": "#ff7f0e",
        "marker": "s",
        "linestyle": "-.",
    },
}
OURS_STYLES = (
    {"color": "#238b45", "marker": "*"},
    {"color": "#6a3d9a", "marker": "P"},
    {"color": "#e6550d", "marker": "X"},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, default=ROOT / "baseline_rd_points.json"
    )
    parser.add_argument(
        "--gvc1d", type=Path, default=ROOT / "gvc1d_rd_points.json"
    )
    parser.add_argument("--ours", type=Path, default=ROOT / "ours_rd_points.json")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def baseline_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for curve in payload["curves"]:
        for point in curve["points"]:
            records.append(
                {
                    "dataset": curve["dataset"],
                    "metric": curve["metric"],
                    "method": curve["method"],
                    "bpp": float(point["bpp"]),
                    "value": float(point["value"]),
                }
            )
    return records


def ours_records(
    payload: dict[str, Any],
    metrics: tuple[tuple[str, str], ...] = LEGACY_METRICS,
) -> list[dict[str, Any]]:
    records = []
    for point in payload["points"]:
        for metric, key in metrics:
            records.append(
                {
                    "dataset": point["dataset"],
                    "metric": metric,
                    "method": point["method"],
                    "label": point["label"],
                    "bpp": float(point["bpp"]),
                    "value": float(point[key]),
                }
            )
    return records


def gvc1d_records(
    payload: dict[str, Any],
    metrics: tuple[tuple[str, str], ...] = LEGACY_METRICS,
) -> list[dict[str, Any]]:
    records = []
    for point in payload["points"]:
        for metric, key in metrics:
            records.append(
                {
                    "dataset": payload["protocol"]["dataset"],
                    "metric": metric,
                    "method": "GVC1D",
                    "bpp": float(point["bpp"]),
                    "value": float(point[key]),
                }
            )
    return records


def manifest_records(
    payload: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Expand archived aggregate points into the plotting record format."""
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported eval manifest schema: {payload.get('schema_version')}")

    required = {"bpp", "psnr", "lpips_vgg", "lpips_alex", "dists", "latency_ms"}
    datasets: list[str] = []
    baseline: list[dict[str, Any]] = []
    ours: list[dict[str, Any]] = []
    for point in payload["points"]:
        dataset = point["dataset"].removesuffix("_96frames")
        model = point["model"]
        aggregate = point["aggregate"]
        missing = required - set(aggregate)
        if missing:
            raise ValueError(f"Incomplete eval point for {model} on {dataset}: {sorted(missing)}")
        if dataset not in datasets:
            datasets.append(dataset)
        record = {"dataset": dataset, "method": model, "bpp": float(aggregate["bpp"])}
        rate_point = point.get("rate_point", {})
        if model == "ARCodec":
            if "lambda" not in rate_point or "checkpoint_step" not in rate_point:
                raise ValueError(f"Incomplete ARCodec rate point: {rate_point}")
            record["label"] = f"lambda={rate_point['lambda']:g}, {int(rate_point['checkpoint_step']) // 1000}k"
            target = ours
        elif model == "GVC1D 32x16":
            record["method"] = "GVC1D"
            target = baseline
        else:
            raise ValueError(f"Unsupported archived model: {model}")
        for metric, key in METRICS:
            metric_record = dict(record)
            metric_record.update({"metric": metric, "value": float(aggregate[key])})
            target.append(metric_record)

    if not datasets or not baseline or not ours:
        raise ValueError("Eval manifest must contain both GVC1D and ARCodec points")
    datasets = [dataset for dataset in ("UVG", "MCL-JCV") if dataset in datasets]
    return datasets, baseline, ours


def validate_protocols(
    baseline: dict[str, Any], ours: dict[str, Any], gvc1d: dict[str, Any]
) -> None:
    base_protocol = baseline["protocol"]
    ours_protocol = ours["protocol"]
    if base_protocol["frames_per_sequence"] != ours_protocol["frames_per_sequence"]:
        raise ValueError("Baseline and ARCodec frame protocols differ")
    if base_protocol["color_space"] != ours_protocol["color_space"]:
        raise ValueError("Baseline and ARCodec color protocols differ")
    if ours_protocol["rate_source"] != "estimated_categorical_cross_entropy_bits":
        raise ValueError("ARCodec points must use estimated categorical rate")
    if not set(ours_protocol["datasets"]) <= set(base_protocol["datasets"]):
        raise ValueError("ARCodec datasets are missing from the baseline data")
    gvc1d_protocol = gvc1d["protocol"]
    if gvc1d_protocol["dataset"] not in ours_protocol["datasets"]:
        raise ValueError("GVC1D dataset is missing from the plotted datasets")
    if gvc1d_protocol["frames_per_sequence"] != ours_protocol["frames_per_sequence"]:
        raise ValueError("GVC1D and ARCodec frame protocols differ")
    if gvc1d_protocol["color_space"] != ours_protocol["color_space"]:
        raise ValueError("GVC1D and ARCodec color protocols differ")
    if gvc1d_protocol["rate_source"] != "measured_bitstream_bits":
        raise ValueError("GVC1D points must use measured bitstream rate")
    gvc1d_required = {"qp", "bpp", "psnr", "latency_ms", "lpips_vgg", "dists"}
    for point in gvc1d["points"]:
        if not gvc1d_required <= set(point):
            raise ValueError(f"Incomplete GVC1D point: {point}")
    required = {
        "dataset",
        "method",
        "label",
        "bpp",
        "psnr",
        "latency_ms",
        "lpips_vgg",
        "dists",
    }
    for point in ours["points"]:
        if not required <= set(point):
            raise ValueError(f"Incomplete ARCodec point: {point}")


def plot(
    datasets: list[str],
    baseline: list[dict[str, Any]],
    ours: list[dict[str, Any]],
    output_dir: Path,
    metrics: tuple[tuple[str, str], ...] = LEGACY_METRICS,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    figure, axes = plt.subplots(
        len(metrics),
        len(datasets),
        figsize=(10.0, 7.8),
        squeeze=False,
        sharey="row",
    )
    ours_methods = sorted({record["method"] for record in ours})
    method_styles = {
        method: style for method, style in zip(ours_methods, cycle(OURS_STYLES))
    }

    for column, dataset in enumerate(datasets):
        for row, (metric, _) in enumerate(metrics):
            axis = axes[row, column]
            for method, style in BASELINE_STYLES.items():
                selected = sorted(
                    (
                        record
                        for record in baseline
                        if record["dataset"] == dataset
                        and record["metric"] == metric
                        and record["method"] == method
                    ),
                    key=lambda record: record["bpp"],
                )
                axis.plot(
                    [record["bpp"] for record in selected],
                    [record["value"] for record in selected],
                    linewidth=2.0,
                    markersize=6.5,
                    markerfacecolor=style["color"],
                    markeredgecolor=style["color"],
                    markeredgewidth=1.0,
                    **style,
                )

            for method in ours_methods:
                selected = sorted(
                    (
                        record
                        for record in ours
                        if record["dataset"] == dataset
                        and record["metric"] == metric
                        and record["method"] == method
                    ),
                    key=lambda record: record["bpp"],
                )
                if not selected:
                    continue
                style = method_styles[method]
                axis.plot(
                    [record["bpp"] for record in selected],
                    [record["value"] for record in selected],
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=1.7,
                    markersize=10,
                    markerfacecolor=style["color"],
                    markeredgecolor=style["color"],
                    markeredgewidth=0.8,
                    zorder=3,
                )
                for record in selected:
                    axis.annotate(
                        record.get("label", ""),
                        (record["bpp"], record["value"]),
                        xytext=(4, 5),
                        textcoords="offset points",
                        fontsize=7.5,
                        color=style["color"],
                    )

            if not any(record["dataset"] == dataset for record in ours):
                axis.text(
                    0.5,
                    0.5,
                    "ARCodec MCL-JCV eval not available",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#555555",
                )

            axis.set_title(dataset)
            axis.set_xlabel("Rate (bpp)")
            axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            if column == 0:
                axis.set_ylabel(metric)
            axis.grid(True, color="#d6d6d6", linestyle="--", linewidth=0.7, alpha=0.8)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.margins(x=0.08, y=0.12)

    legend = [
        Line2D(
            [],
            [],
            label=method,
            linewidth=2.0,
            markersize=6.5,
            markerfacecolor=style["color"],
            markeredgecolor=style["color"],
            **style,
        )
        for method, style in BASELINE_STYLES.items()
    ]
    legend.extend(
        Line2D(
            [],
            [],
            label=method,
            color=method_styles[method]["color"],
            marker=method_styles[method]["marker"],
            linewidth=1.7,
            markersize=9,
            markerfacecolor=method_styles[method]["color"],
            markeredgecolor=method_styles[method]["color"],
        )
        for method in ours_methods
    )
    figure.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=len(legend),
        frameon=False,
    )
    figure.suptitle("Perceptual rate-distortion", y=0.985, fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "rd_curves.png", dpi=300, facecolor="white")
    figure.savefig(output_dir / "rd_curves.pdf", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.manifest.exists():
        datasets, archived_baseline, ours = manifest_records(read_json(args.manifest))
        baseline = baseline_records(read_json(args.baseline)) + archived_baseline
        plot(datasets, baseline, ours, args.output_dir, metrics=METRICS)
    else:
        baseline_payload = read_json(args.baseline)
        gvc1d_payload = read_json(args.gvc1d)
        ours_payload = read_json(args.ours)
        validate_protocols(baseline_payload, ours_payload, gvc1d_payload)
        datasets = list(ours_payload["protocol"]["datasets"])
        plot(
            datasets,
            baseline_records(baseline_payload) + gvc1d_records(gvc1d_payload),
            ours_records(ours_payload),
            args.output_dir,
        )
    print(args.output_dir / "rd_curves.png")
    print(args.output_dir / "rd_curves.pdf")


if __name__ == "__main__":
    main()
