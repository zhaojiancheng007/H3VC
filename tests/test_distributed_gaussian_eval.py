import math
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from training.evaluation_categorical import (
    HEVC_B_SEQUENCES,
    MCL_JCV_SEQUENCES,
    UVG_SEQUENCES,
)
from scripts.stage2_cosine import evaluation_command
from training.evaluation_categorical_distributed import (
    SequenceJob,
    build_benchmark_jobs,
    evaluate_shard,
    merge_evaluation_shards,
    replace_reconstruction_archive,
    save_reconstruction,
    shard_jobs,
)


def test_all_stages_route_four_rank_distributed_gaussian_evaluation() -> None:
    args = Namespace(
        config="configs/stage3_lambda18_cosine.yaml",
        uvg_root="uvg",
        mcl_jcv_root="mcl",
        hevc_b_root="hevc",
        wan_repo="wan",
        wan_checkpoint="wan.pt",
        cosmos_repo="cosmos",
        cosmos_encoder="encoder.jit",
        cosmos_decoder="decoder.jit",
        cosmos_cv888_encoder="encoder888.jit",
        cosmos_cv888_decoder="decoder888.jit",
        minimax_h3_model_dir="h3",
    )
    expected = {
        1: "evaluate_stage1_tokenizer_gaussian_distributed.py",
        2: "evaluate_stage2_cosine_gaussian_distributed.py",
        3: "evaluate_stage3_cosine_gaussian_distributed.py",
    }

    for stage, evaluator in expected.items():
        command = evaluation_command(
            args, stage, Path("checkpoint.pt"), Path("evaluation"), 1, 4, Path("/project")
        )
        assert Path(command[1]).name == evaluator
        assert command[command.index("--rank") + 1] == "1"
        assert command[command.index("--world-size") + 1] == "4"
        if stage == 3:
            assert command[command.index("--recon-dir") + 1] == str(
                Path("evaluation/recon").resolve()
            )
        else:
            assert "--recon-dir" not in command
        assert ("--config" in command) is (stage in (2, 3))


def test_stage1_distributed_shard_uses_raw_index_bits() -> None:
    job = SequenceJob("uvg", UVG_SEQUENCES[0], Path("uvg"))
    tokenizer = SimpleNamespace(
        quantizer=SimpleNamespace(embed=torch.empty(4096, 24))
    )
    with patch(
        "training.evaluation_categorical_distributed.evaluate_sequence_job",
        return_value=_sequence_report(job, 0),
    ):
        shard = evaluate_shard(
            (job,), tokenizer, None, "minimax_h3", None, None, None, None,
            None,
            rank=0, world_size=1,
        )

    assert shard["protocol"]["rate"] == (
        "12 raw bits per index without entropy coding"
    )


def test_reconstruction_is_saved_as_numbered_png_frames(tmp_path: Path) -> None:
    reconstruction = torch.tensor(
        [
            [[[0.0, 1.0]], [[0.5, 0.25]], [[1.0, 0.0]]],
            [[[1.0, 0.0]], [[0.25, 0.5]], [[0.0, 1.0]]],
        ]
    )

    save_reconstruction(reconstruction, tmp_path / "uvg" / "Beauty")

    assert sorted(path.name for path in (tmp_path / "uvg" / "Beauty").iterdir()) == [
        "000001.png",
        "000002.png",
    ]


def test_reconstruction_archive_replaces_previous_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "step_0025000" / "recon"
    canonical = tmp_path / "recon"
    source.mkdir(parents=True)
    canonical.mkdir()
    (source / "new.txt").write_text("25k")
    (canonical / "old.txt").write_text("20k")

    replace_reconstruction_archive(source, canonical)

    assert not source.exists()
    assert [path.name for path in canonical.iterdir()] == ["new.txt"]


def test_all_benchmark_sequences_are_balanced_across_four_ranks() -> None:
    jobs = build_benchmark_jobs(Path("uvg"), Path("mcl"), Path("hevc"))
    shards = [shard_jobs(jobs, rank, 4) for rank in range(4)]

    assert len(jobs) == 42
    assert [len(shard) for shard in shards] == [11, 11, 10, 10]
    assert {job for shard in shards for job in shard} == set(jobs)
    assert sum(len(shard) for shard in shards) == len(jobs)


def _sequence_report(job: SequenceJob, index: int) -> dict:
    bits = float(index + 8)
    return {
        "dataset": job.dataset,
        "sequence": job.sequence,
        "frames": 2,
        "rate_frames": 2,
        "resolution": [2, 2],
        "psnr": 10.0,
        "bpp": bits / 8.0,
        "lpips": float(index),
        "lpips_vgg": float(index),
        "lpips_alex": float(index + 1),
        "dists": float(index + 2),
        "latency_ms": float(index + 3),
        "bits": bits,
        "squared_error": 4.0,
        "value_count": 24,
    }


def _synthetic_shards() -> list[dict]:
    jobs = build_benchmark_jobs(Path("uvg"), Path("mcl"), Path("hevc"))
    protocol = {
        "model_input_frames": 107,
        "metric_frames": 96,
        "bpp_denominator_frames": 96,
        "rate": "estimated categorical cross-entropy bits",
    }
    shards = [
        {
            "schema_version": 1,
            "rank": rank,
            "world_size": 4,
            "tokenizer": "minimax_h3",
            "checkpoint_step": 5000,
            "protocol": protocol,
            "sequences": [],
        }
        for rank in range(4)
    ]
    for index, job in enumerate(jobs):
        shards[index % 4]["sequences"].append(_sequence_report(job, index))
    return shards


def test_shards_merge_to_original_dataset_reports() -> None:
    outputs = merge_evaluation_shards(_synthetic_shards())

    assert set(outputs) == {"uvg", "mcl_jcv", "hevc_b"}
    assert outputs["uvg"]["protocol"]["sequences"] == list(UVG_SEQUENCES)
    assert outputs["mcl_jcv"]["protocol"]["sequences"] == list(
        MCL_JCV_SEQUENCES
    )
    assert outputs["hevc_b"]["protocol"]["sequences"] == list(
        HEVC_B_SEQUENCES
    )
    assert outputs["uvg"]["checkpoint_step"] == 5000

    mcl_reports = outputs["mcl_jcv"]["sequences"]
    assert all("dataset" not in report for report in mcl_reports)
    expected_bits = sum(float(index + 8) for index in range(7, 37))
    assert outputs["mcl_jcv"]["aggregate"]["bpp"] == pytest.approx(
        expected_bits / (30 * 2 * 2 * 2)
    )
    assert outputs["mcl_jcv"]["aggregate"]["psnr"] == pytest.approx(
        -10.0 * math.log10(4.0 / 24)
    )
    assert {
        "psnr",
        "bpp",
        "latency_ms",
        "lpips",
        "lpips_vgg",
        "lpips_alex",
        "dists",
    } <= set(outputs["mcl_jcv"]["aggregate"])


def test_merge_rejects_incomplete_sequence_coverage() -> None:
    shards = _synthetic_shards()
    shards[-1]["sequences"].pop()

    with pytest.raises(ValueError, match="coverage changed"):
        merge_evaluation_shards(shards)
