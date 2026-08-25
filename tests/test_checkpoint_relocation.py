from pathlib import Path

import pytest
import torch

from scripts.stage3 import load_stage3_checkpoint
from training.checkpoint_paths import assert_checkpoint_reference
from training.config_stage3_cosine import load_stage3_config, stage3_config_dict
from training.distributed_runtime import IntervalProgress


def test_checkpoint_relocation_requires_explicit_permission(tmp_path: Path) -> None:
    archived_checkpoint = tmp_path / "archive" / "stage1.pt"
    archived_checkpoint.parent.mkdir()
    archived_checkpoint.touch()
    with pytest.raises(ValueError, match="allow-checkpoint-relocation"):
        assert_checkpoint_reference(
            "/deleted/container/stage1.pt",
            archived_checkpoint,
            label="Stage1 checkpoint",
            allow_relocation=False,
        )

    assert assert_checkpoint_reference(
        "/deleted/container/stage1.pt",
        archived_checkpoint,
        label="Stage1 checkpoint",
        allow_relocation=True,
    ) is True


def test_progress_prints_once_every_ten_steps(capsys):
    progress = IntervalProgress(
        total=60_000,
        initial=15_000,
        desc="Stage3 minimax_h3 joint",
        interval=10,
    )

    for _ in range(9):
        progress.update()
        progress.set_postfix(bpp="0.0050", lpips="0.100", psnr="32.00")
    assert capsys.readouterr().out == ""

    progress.update()
    progress.set_postfix(bpp="0.0050", lpips="0.100", psnr="32.00")

    assert capsys.readouterr().out == (
        "Stage3 minimax_h3 joint: 15010/60000 "
        "bpp=0.0050 lpips=0.100 psnr=32.00\n"
    )


@pytest.mark.parametrize("version", (2, 3))
def test_stage3_checkpoint_loader_accepts_legacy_and_canonical_fields(
    tmp_path: Path, version: int
) -> None:
    payload = {
        "version": version,
        "stage": 3,
        "step": 15_000,
        "tokenizer": "minimax_h3",
        "tokenizer_checkpoint": "/archive/stage1.pt",
        "entropy_checkpoint": "/archive/stage2.pt",
        "config": {},
        "model": {},
        "optimizer": {},
        "training_state": {},
        "rng_states": [],
    }
    if version == 2:
        payload["discriminator"] = {}
        payload["discriminator_optimizer"] = {}
    checkpoint = tmp_path / f"stage3_v{version}.pt"
    torch.save(payload, checkpoint)

    assert load_stage3_checkpoint(checkpoint)["version"] == version


def test_legacy_zero_gan_config_normalizes_to_canonical(tmp_path: Path) -> None:
    canonical = Path(__file__).resolve().parents[1] / "configs/stage3.yaml"
    text = canonical.read_text(encoding="utf-8")
    text = text.replace(
        "optimizer:\n",
        "optimizer:\n  discriminator_learning_rate: 0.0001\n",
    ).replace(
        "  temperature: 0.01\n",
        "  temperature: 0.01\n  gan_start_step: 20000\n  gan_weight: 0.0\n",
    )
    legacy = tmp_path / "legacy_stage3.yaml"
    legacy.write_text(text, encoding="utf-8")

    assert stage3_config_dict(load_stage3_config(legacy)) == stage3_config_dict(
        load_stage3_config(canonical)
    )
