#!/usr/bin/env python3
"""Build multi-length scene-sampled metadata for extracted OpenVid MP4 files."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path, PurePosixPath
import random
import time
from typing import Any
from zipfile import ZipFile

from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector


DEFAULT_SEQUENCE_LENGTHS = (5, 13, 25, 33, 49, 65, 81, 97)
INCOMPLETE_MARKER = ".incomplete"
MANIFEST_NAME = "run_manifest.json"
METADATA_NAME = "metadata.jsonl"
REJECTED_NAME = "rejected.jsonl"
REPORT_NAME = "processing_report.json"


@dataclass(frozen=True)
class Candidate:
    archive: str
    member: str
    source_bytes: int
    source_crc32: int
    video_path: str


@dataclass(frozen=True)
class SceneSettings:
    sequence_lengths: tuple[int, ...]
    width: int
    height: int
    scene_threshold: float
    seed: int

    @property
    def min_scene_len(self) -> int:
        return self.sequence_lengths[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run scene detection over every extracted OpenVid MP4 and record "
            "deterministic samples for multiple Cosmos-compatible lengths"
        )
    )
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sequence-lengths",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEQUENCE_LENGTHS),
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--scene-threshold", type=float, default=27.0)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--stop-after-sources", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} is blank")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} must contain a JSON object")
            values.append(value)
    return values


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def stable_candidates(
    archives: list[Path], video_root: Path, seed: int
) -> list[Candidate]:
    if not video_root.is_dir():
        raise FileNotFoundError(video_root)
    candidates: list[Candidate] = []
    seen_members: set[str] = set()
    for archive_arg in archives:
        archive = archive_arg.resolve()
        if not archive.is_file():
            raise FileNotFoundError(archive)
        with ZipFile(archive) as zip_file:
            for info in zip_file.infolist():
                if info.is_dir():
                    continue
                member_path = PurePosixPath(info.filename)
                if (
                    member_path.is_absolute()
                    or ".." in member_path.parts
                    or member_path.suffix.lower() != ".mp4"
                ):
                    raise ValueError(
                        f"Archive {archive} has invalid MP4 member {info.filename!r}"
                    )
                if info.filename in seen_members:
                    raise ValueError(
                        f"Extracted layout cannot represent duplicate member "
                        f"{info.filename!r}"
                    )
                seen_members.add(info.filename)
                video_path = video_root.joinpath(*member_path.parts)
                if not video_path.is_file():
                    raise FileNotFoundError(video_path)
                if video_path.stat().st_size != info.file_size:
                    raise IOError(
                        f"Extracted size mismatch for {archive.name}:{info.filename}"
                    )
                candidates.append(
                    Candidate(
                        archive=archive.name,
                        member=info.filename,
                        source_bytes=info.file_size,
                        source_crc32=info.CRC,
                        video_path=str(video_path),
                    )
                )
    candidates.sort(key=lambda item: (item.archive, item.member))
    random.Random(seed).shuffle(candidates)
    return candidates


def select_samples(
    scenes: list[tuple[int, int]],
    *,
    archive: str,
    member: str,
    settings: SceneSettings,
) -> dict[str, dict[str, int]]:
    samples: dict[str, dict[str, int]] = {}
    for frames in settings.sequence_lengths:
        eligible = [(start, end) for start, end in scenes if end - start >= frames]
        if not eligible:
            continue
        source_rng = random.Random(
            f"{settings.seed}:{archive}:{member}:{frames}"
        )
        scene_start, scene_end = source_rng.choice(eligible)
        sample_start = source_rng.randint(scene_start, scene_end - frames)
        samples[str(frames)] = {
            "frames": frames,
            "scene_start_frame": scene_start,
            "scene_end_frame_exclusive": scene_end,
            "sample_start_frame": sample_start,
            "sample_end_frame_inclusive": sample_start + frames - 1,
        }
    return samples


def analyze_candidate(
    candidate: Candidate, *, settings: SceneSettings
) -> dict[str, Any]:
    video = open_video(candidate.video_path)
    frame_size = tuple(map(int, video.frame_size))
    frame_rate = float(video.frame_rate)
    if frame_size != (settings.width, settings.height):
        return {
            "accepted": False,
            "record": {
                "archive": candidate.archive,
                "member": candidate.member,
                "reason": "resolution_mismatch",
                "width": frame_size[0],
                "height": frame_size[1],
            },
        }

    manager = SceneManager()
    manager.add_detector(
        ContentDetector(
            threshold=settings.scene_threshold,
            min_scene_len=settings.min_scene_len,
        )
    )
    manager.detect_scenes(video, show_progress=False)
    scenes = [
        (start.get_frames(), end.get_frames())
        for start, end in manager.get_scene_list(start_in_scene=True)
    ]
    samples = select_samples(
        scenes,
        archive=candidate.archive,
        member=candidate.member,
        settings=settings,
    )
    if not samples:
        lengths = [end - start for start, end in scenes]
        return {
            "accepted": False,
            "record": {
                "archive": candidate.archive,
                "member": candidate.member,
                "reason": "no_scene_with_minimum_frames",
                "scene_count": len(scenes),
                "longest_scene_frames": max(lengths, default=0),
            },
        }

    available = [
        frames for frames in settings.sequence_lengths if str(frames) in samples
    ]
    return {
        "accepted": True,
        "record": {
            "archive": candidate.archive,
            "member": candidate.member,
            "source_bytes": candidate.source_bytes,
            "source_crc32": f"{candidate.source_crc32:08x}",
            "source_frame_rate": frame_rate,
            "width": settings.width,
            "height": settings.height,
            "scene_detector_min_scene_len": settings.min_scene_len,
            "scenes": [
                {
                    "start_frame": start,
                    "end_frame_exclusive": end,
                }
                for start, end in scenes
            ],
            "available_sequence_lengths": available,
            "samples": samples,
        },
    }


def source_key(record: dict[str, Any]) -> tuple[str, str]:
    archive = record.get("archive")
    member = record.get("member")
    if not isinstance(archive, str) or not archive:
        raise ValueError(f"Record has invalid archive: {record}")
    if not isinstance(member, str) or not member:
        raise ValueError(f"Record has invalid member: {record}")
    return archive, member


def validate_state(
    metadata: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    *,
    candidate_keys: set[tuple[str, str]],
    settings: SceneSettings,
) -> set[tuple[str, str]]:
    accepted_keys: set[tuple[str, str]] = set()
    for index, record in enumerate(metadata, start=1):
        if record.get("sequence") != f"{index:06d}":
            raise ValueError(f"Metadata sequence {index} is not contiguous")
        key = source_key(record)
        if key in accepted_keys:
            raise ValueError(f"Duplicate accepted source: {key}")
        if key not in candidate_keys:
            raise ValueError(f"Accepted source is not a candidate: {key}")
        if (
            int(record.get("width", -1)) != settings.width
            or int(record.get("height", -1)) != settings.height
        ):
            raise ValueError(f"Accepted source has incompatible resolution: {key}")

        scenes = record.get("scenes")
        samples = record.get("samples")
        available = record.get("available_sequence_lengths")
        if not isinstance(scenes, list) or not scenes:
            raise ValueError(f"Accepted source has no scenes: {key}")
        if not isinstance(samples, dict) or not samples:
            raise ValueError(f"Accepted source has no samples: {key}")
        expected_available = [
            frames
            for frames in settings.sequence_lengths
            if str(frames) in samples
        ]
        if available != expected_available:
            raise ValueError(f"Accepted source has inconsistent lengths: {key}")

        scene_bounds = {
            (int(scene["start_frame"]), int(scene["end_frame_exclusive"]))
            for scene in scenes
        }
        for frames in expected_available:
            sample = samples[str(frames)]
            if int(sample.get("frames", -1)) != frames:
                raise ValueError(f"Sample {frames} has wrong length: {key}")
            scene_start = int(sample["scene_start_frame"])
            scene_end = int(sample["scene_end_frame_exclusive"])
            sample_start = int(sample["sample_start_frame"])
            sample_end = int(sample["sample_end_frame_inclusive"])
            if (scene_start, scene_end) not in scene_bounds:
                raise ValueError(f"Sample {frames} references an unknown scene: {key}")
            if (
                sample_end != sample_start + frames - 1
                or sample_start < scene_start
                or sample_end >= scene_end
            ):
                raise ValueError(f"Sample {frames} has invalid bounds: {key}")
        accepted_keys.add(key)

    rejected_keys: set[tuple[str, str]] = set()
    for record in rejected:
        key = source_key(record)
        if key in rejected_keys:
            raise ValueError(f"Duplicate rejected source: {key}")
        if key not in candidate_keys:
            raise ValueError(f"Rejected source is not a candidate: {key}")
        if not isinstance(record.get("reason"), str):
            raise ValueError(f"Rejected source has no reason: {key}")
        rejected_keys.add(key)

    overlap = accepted_keys & rejected_keys
    if overlap:
        raise ValueError(f"Sources are both accepted and rejected: {sorted(overlap)[:5]}")
    return accepted_keys | rejected_keys


def build_manifest(
    args: argparse.Namespace,
    candidates: list[Candidate],
    settings: SceneSettings,
) -> dict[str, Any]:
    serialized_settings = asdict(settings)
    serialized_settings["sequence_lengths"] = list(settings.sequence_lengths)
    return {
        "schema_version": 1,
        "archives": [str(path.resolve()) for path in args.archives],
        "video_root": str(args.video_root.resolve()),
        "candidate_mp4_files": len(candidates),
        "settings": serialized_settings,
        "scene_detector": "PySceneDetect ContentDetector",
        "sampling_rng": "per-source seed: global_seed:archive:member:length",
    }


def initialize_output(
    args: argparse.Namespace,
    candidates: list[Candidate],
    settings: SceneSettings,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    metadata_path = args.output / METADATA_NAME
    rejected_path = args.output / REJECTED_NAME
    manifest_path = args.output / MANIFEST_NAME
    marker_path = args.output / INCOMPLETE_MARKER
    expected_manifest = build_manifest(args, candidates, settings)

    if args.resume:
        if not args.output.is_dir() or not marker_path.is_file():
            raise RuntimeError("Resume requires an output directory with .incomplete")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest != expected_manifest:
            raise ValueError("Resume settings do not match run_manifest.json")
        return read_jsonl(metadata_path), read_jsonl(rejected_path), manifest

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path.touch()
    rejected_path.touch()
    write_json(manifest_path, expected_manifest)
    write_json(
        marker_path,
        {
            "schema_version": 1,
            "status": "incomplete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return [], [], expected_manifest


def main() -> None:
    args = parse_args()
    sequence_lengths = tuple(args.sequence_lengths)
    if (
        not sequence_lengths
        or tuple(sorted(set(sequence_lengths))) != sequence_lengths
        or any(frames <= 0 or (frames - 1) % 4 for frames in sequence_lengths)
    ):
        raise ValueError(
            "Sequence lengths must be unique increasing positive values of 4k+1"
        )
    positive = {
        "width": args.width,
        "height": args.height,
        "workers": args.workers,
        "progress_interval": args.progress_interval,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"Arguments must be positive: {invalid}")
    if args.stop_after_sources is not None and args.stop_after_sources <= 0:
        raise ValueError("--stop-after-sources must be positive")

    settings = SceneSettings(
        sequence_lengths=sequence_lengths,
        width=args.width,
        height=args.height,
        scene_threshold=args.scene_threshold,
        seed=args.seed,
    )
    candidates = stable_candidates(args.archives, args.video_root, args.seed)
    candidate_keys = {(item.archive, item.member) for item in candidates}
    if len(candidate_keys) != len(candidates):
        raise ValueError("Candidate source keys are not unique")

    metadata, rejected, manifest = initialize_output(args, candidates, settings)
    processed = validate_state(
        metadata,
        rejected,
        candidate_keys=candidate_keys,
        settings=settings,
    )
    remaining = [
        candidate
        for candidate in candidates
        if (candidate.archive, candidate.member) not in processed
    ]
    selected = remaining
    if args.stop_after_sources is not None:
        selected = remaining[: args.stop_after_sources]

    accepted_count = len(metadata)
    rejected_count = len(rejected)
    rejected_counts = Counter(record["reason"] for record in rejected)
    started = time.perf_counter()
    worker = partial(analyze_candidate, settings=settings)

    with (args.output / METADATA_NAME).open(
        "a", encoding="utf-8", buffering=1
    ) as metadata_file, (args.output / REJECTED_NAME).open(
        "a", encoding="utf-8", buffering=1
    ) as rejected_file:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for invocation_index, result in enumerate(
                executor.map(worker, selected, chunksize=1), start=1
            ):
                record = result["record"]
                if result["accepted"]:
                    accepted_count += 1
                    record["sequence"] = f"{accepted_count:06d}"
                    metadata_file.write(
                        json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
                    )
                else:
                    rejected_count += 1
                    rejected_counts[record["reason"]] += 1
                    rejected_file.write(
                        json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
                    )

                processed_count = accepted_count + rejected_count
                if (
                    invocation_index == 1
                    or invocation_index % args.progress_interval == 0
                    or invocation_index == len(selected)
                ):
                    elapsed = time.perf_counter() - started
                    print(
                        json.dumps(
                            {
                                "accepted": accepted_count,
                                "candidates": len(candidates),
                                "elapsed_seconds": elapsed,
                                "processed": processed_count,
                                "rejected": rejected_count,
                                "remaining": len(candidates) - processed_count,
                                "sources_per_second": invocation_index / elapsed,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    processed_count = accepted_count + rejected_count
    if processed_count < len(candidates):
        print(
            json.dumps(
                {
                    "event": "finished_invocation",
                    "status": "incomplete",
                    "processed": processed_count,
                    "remaining": len(candidates) - processed_count,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    if processed_count != len(candidates):
        raise RuntimeError(
            f"Processed {processed_count} sources for {len(candidates)} candidates"
        )

    metadata_path = args.output / METADATA_NAME
    rejected_path = args.output / REJECTED_NAME
    final_metadata = read_jsonl(metadata_path)
    final_rejected = read_jsonl(rejected_path)
    final_processed = validate_state(
        final_metadata,
        final_rejected,
        candidate_keys=candidate_keys,
        settings=settings,
    )
    if len(final_processed) != len(candidates):
        raise RuntimeError("Final metadata does not cover every candidate")

    available_counts = {
        str(frames): sum(
            frames in record["available_sequence_lengths"]
            for record in final_metadata
        )
        for frames in settings.sequence_lengths
    }
    report = {
        **manifest,
        "status": "pass",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "accepted_videos": len(final_metadata),
        "rejected_videos": len(final_rejected),
        "rejected": dict(sorted(rejected_counts.items())),
        "available_videos_by_sequence_length": available_counts,
        "processed_sources": len(final_processed),
    }
    write_json(args.output / REPORT_NAME, report)
    (args.output / INCOMPLETE_MARKER).unlink()
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
