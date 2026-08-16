import argparse
import json
from pathlib import Path

from scripts.prepare_openvid_mp4_multilength_scenes import (
    Candidate,
    SceneSettings,
    build_manifest,
    select_samples,
)


def _settings() -> SceneSettings:
    return SceneSettings(
        sequence_lengths=(5, 13, 25, 33, 49, 65, 81, 97),
        width=1920,
        height=1080,
        scene_threshold=27.0,
        seed=20260726,
    )


def test_select_samples_records_every_supported_length() -> None:
    scenes = [(0, 4), (10, 107)]
    settings = _settings()

    samples = select_samples(
        scenes,
        archive="part0.zip",
        member="sample.mp4",
        settings=settings,
    )

    assert list(samples) == [str(frames) for frames in settings.sequence_lengths]
    assert samples == select_samples(
        scenes,
        archive="part0.zip",
        member="sample.mp4",
        settings=settings,
    )
    for frames in settings.sequence_lengths:
        sample = samples[str(frames)]
        assert sample["frames"] == frames
        assert sample["scene_start_frame"] == 10
        assert sample["scene_end_frame_exclusive"] == 107
        assert sample["sample_end_frame_inclusive"] == (
            sample["sample_start_frame"] + frames - 1
        )
        assert sample["sample_end_frame_inclusive"] < 107


def test_build_manifest_is_json_round_trip_stable(tmp_path: Path) -> None:
    archive = tmp_path / "part0.zip"
    video_root = tmp_path / "videos"
    args = argparse.Namespace(archives=[archive], video_root=video_root)
    candidates = [
        Candidate(
            archive=archive.name,
            member="sample.mp4",
            source_bytes=123,
            source_crc32=456,
            video_path=str(video_root / "sample.mp4"),
        )
    ]

    manifest = build_manifest(args, candidates, _settings())

    assert manifest["settings"]["sequence_lengths"] == [
        5,
        13,
        25,
        33,
        49,
        65,
        81,
        97,
    ]
    assert json.loads(json.dumps(manifest)) == manifest
