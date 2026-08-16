import ast
import json
from pathlib import Path

import pytest
import torch

from entropy_model.config_categorical import CategoricalEntropyModelConfig
from entropy_model.model_categorical import CategoricalEntropyModel
from scripts.stage2 import (
    keep_best_bpp_checkpoint,
    scheduler_scale,
)
from scripts.stage3 import load_stage2_checkpoint, rate_parameters
from tokenizer.stage3_vq import relaxed_quantize
from training import evaluation_categorical
from training.config_stage2 import learning_rate_at_step, load_stage2_config
from training.config_stage3 import learning_rates_at_step, load_stage3_config


ROOT = Path(__file__).resolve().parents[1]


def tiny_entropy_model() -> CategoricalEntropyModel:
    return CategoricalEntropyModel(
        CategoricalEntropyModelConfig(
            window_size=4,
            d_model=24,
            num_layers=1,
            num_heads=4,
            ffn_dim=48,
            max_latent_frames=2,
        )
    )


def test_stage2_predicts_one_index_distribution_per_position() -> None:
    model = tiny_entropy_model()
    features = torch.randn(1, 16, 2, 4, 4)
    indices = torch.randint(0, 4096, (1, 2, 4, 4))

    output = model(features, indices)

    assert output.logits.shape == (1, 8, 4, 4096)
    assert torch.allclose(output.bits_per_index, torch.tensor(12.0))
    output.bits_per_index.backward()
    assert model.output_head.weight.grad is not None


def test_stage3_reuses_one_distance_matrix_for_hard_and_soft_targets() -> None:
    torch.manual_seed(7)
    projected = torch.randn(1, 16, 1, 4, 4, requires_grad=True)
    codebook = torch.randn(4096, 16)

    quantized = relaxed_quantize(projected, codebook, temperature=0.1)

    flat = projected.detach().permute(0, 2, 3, 4, 1).reshape(-1, 16)
    reference = torch.cdist(
        flat, codebook, p=2, compute_mode="donot_use_mm_for_euclid_dist"
    ).argmin(dim=1).reshape(1, 1, 4, 4)

    assert torch.equal(quantized.indices, reference)
    assert quantized.indices.shape == (1, 1, 4, 4)
    assert quantized.target_probabilities.shape == (1, 1, 4, 4, 4096)
    torch.testing.assert_close(
        quantized.target_probabilities.sum(dim=-1),
        torch.ones(1, 1, 4, 4),
    )
    objective = (
        quantized.quantized.square().mean()
        + quantized.target_probabilities[..., 0].mean()
    )
    objective.backward()
    assert projected.grad is not None
    assert torch.isfinite(projected.grad).all()


def test_canonical_configs_use_4096x24_without_gradient_accumulation() -> None:
    stage2 = load_stage2_config(ROOT / "configs/stage2.yaml")
    stage3 = load_stage3_config(ROOT / "configs/stage3.yaml")

    assert stage2.model.codebook_size == stage3.model.codebook_size == 4096
    assert stage2.model.codebook_dim == stage3.model.codebook_dim == 24
    assert not hasattr(stage2.train, "gradient_accumulation")
    assert not hasattr(stage3.train, "gradient_accumulation")
    assert rate_parameters(0, stage3) == (7.2, 0.01)
    assert rate_parameters(60_000, stage3) == (7.2, 0.01)
    assert stage3.loss.gan_start_step == 20000
    assert stage3.loss.gan_weight == 0.0


def test_h3_stage2_multistep_schedule_and_best_policy() -> None:
    config = load_stage2_config(ROOT / "configs/stage2.yaml")

    assert config.train.max_steps == 60_000
    assert config.train.checkpoint_interval == 5_000
    assert config.train.checkpoint_policy == "best_bpp"
    expected = (
        (0, 5e-5),
        (14_999, 5e-5),
        (15_000, 1e-5),
        (29_999, 1e-5),
        (30_000, 5e-6),
        (44_999, 5e-6),
        (45_000, 1e-6),
        (60_000, 1e-6),
    )
    for step, rate in expected:
        assert learning_rate_at_step(config.optimizer, step) == rate
    assert scheduler_scale(config, 14_999) == pytest.approx(0.2)
    assert scheduler_scale(config, 29_999) == pytest.approx(0.1)
    assert scheduler_scale(config, 44_999) == pytest.approx(0.02)


def test_h3_stage3_uses_fixed_rate_and_multistep_learning_rates() -> None:
    config = load_stage3_config(ROOT / "configs/stage3.yaml")

    assert config.train.max_steps == 60_000
    assert config.train.checkpoint_interval == 5_000
    for step in (0, 15_000, 30_000, 45_000, 60_000):
        assert rate_parameters(step, config) == (7.2, 0.01)
    expected = (
        (0, 1e-5, 5e-5),
        (14_999, 1e-5, 5e-5),
        (15_000, 5e-6, 1e-5),
        (29_999, 5e-6, 1e-5),
        (30_000, 1e-6, 5e-6),
        (44_999, 1e-6, 5e-6),
        (45_000, 5e-7, 1e-6),
        (60_000, 5e-7, 1e-6),
    )
    for step, tokenizer_rate, entropy_rate in expected:
        assert learning_rates_at_step(config.optimizer, step) == (
            tokenizer_rate,
            entropy_rate,
        )


def test_stage3_accepts_version_2_stage2_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "stage2.pt"
    payload = {
        "version": 2,
        "stage": 2,
        "step": 60_000,
        "tokenizer": "minimax_h3",
        "tokenizer_checkpoint": "stage1.pt",
        "config": {},
        "model": {},
        "optimizer": {},
        "scheduler": {},
        "training_state": {},
        "rng_states": [],
    }
    torch.save(payload, checkpoint)

    assert load_stage2_checkpoint(checkpoint)["version"] == 2


def test_stage2_best_bpp_policy_keeps_only_best_checkpoint(tmp_path: Path) -> None:
    candidate = tmp_path / ".candidate.pt"
    evaluation = tmp_path / "evaluation.json"
    best = tmp_path / "checkpoint_best.pt"
    record = tmp_path / "best_checkpoint.json"

    def write_candidate(step: int, bpp: float) -> None:
        candidate.write_bytes(f"checkpoint-{step}".encode())
        evaluation.write_text(
            json.dumps(
                {
                    "checkpoint_step": step,
                    "aggregate": {
                        "psnr": 30.0,
                        "bpp": bpp,
                        "lpips": 0.1,
                        "dists": 0.1,
                        "latency_ms": 1.0,
                    },
                }
            ),
            encoding="utf-8",
        )

    write_candidate(5_000, 0.02)
    assert keep_best_bpp_checkpoint(candidate, evaluation, best, record)
    assert best.read_bytes() == b"checkpoint-5000"

    write_candidate(10_000, 0.03)
    assert not keep_best_bpp_checkpoint(candidate, evaluation, best, record)
    assert not candidate.exists()
    assert best.read_bytes() == b"checkpoint-5000"

    write_candidate(15_000, 0.01)
    assert keep_best_bpp_checkpoint(candidate, evaluation, best, record)
    assert best.read_bytes() == b"checkpoint-15000"
    assert json.loads(record.read_text())["checkpoint_step"] == 15_000


def test_active_training_scripts_do_not_import_other_scripts() -> None:
    for name in (
        "stage1.py",
        "stage2.py",
        "stage3.py",
    ):
        path = ROOT / "scripts" / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_scripts = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("scripts.")
        }
        assert imported_scripts == set()


def test_h3_uvg_rate_uses_one_path_for_raw_and_entropy_bits(monkeypatch) -> None:
    class Quantizer:
        embed = torch.zeros(4096, 24)

        @staticmethod
        def _nearest_indices(flat: torch.Tensor) -> torch.Tensor:
            return torch.zeros(flat.shape[0], dtype=torch.long)

    class Tokenizer:
        quantizer = Quantizer()
        pre_quant = staticmethod(lambda latent: latent)
        post_quant = staticmethod(lambda latent: latent)

        def eval(self):
            return self

        @staticmethod
        def encode(video: torch.Tensor) -> torch.Tensor:
            assert video.shape == (1, 3, 107, 16, 16)
            return torch.zeros(1, 24, 1, 1, 1)

        @staticmethod
        def decode(latent: torch.Tensor) -> torch.Tensor:
            return torch.zeros(1, 3, 107, 16, 16)

    class EntropyModel:
        def eval(self):
            return self

    monkeypatch.setattr(evaluation_categorical, "UVG_SEQUENCES", ("test",))
    monkeypatch.setattr(
        evaluation_categorical,
        "load_png_sequence",
        lambda *_args, **_kwargs: (torch.zeros(96, 3, 16, 16), None),
    )
    monkeypatch.setattr(
        evaluation_categorical,
        "_estimated_bits",
        lambda *_args: torch.tensor(96.0),
    )
    monkeypatch.setattr(evaluation_categorical, "sequence_lpips", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(evaluation_categorical, "_sequence_dists", lambda *_args: 0.0)
    monkeypatch.setattr(evaluation_categorical.torch.cuda, "synchronize", lambda *_args: None)

    report = evaluation_categorical.evaluate_uvg_v2(
        Tokenizer(), EntropyModel(), "minimax_h3", "/unused", None, None,
        torch.device("cpu"),
    )

    raw_report = evaluation_categorical.evaluate_uvg_v2(
        Tokenizer(), None, "minimax_h3", "/unused", None, None,
        torch.device("cpu"),
    )

    for result in (report, raw_report):
        assert result["protocol"]["model_input_frames"] == 107
        assert result["protocol"]["bpp_denominator_frames"] == 96
    assert report["aggregate"]["bpp"] == 1.0 / 256.0
    assert raw_report["aggregate"]["bpp"] == 1.0 / 2048.0
