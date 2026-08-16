import ast
from pathlib import Path

import torch

from entropy_model.config_categorical import CategoricalEntropyModelConfig
from entropy_model.model_categorical import CategoricalEntropyModel
from scripts.stage3_jointtraining import rate_parameters
from tokenizer.stage3_vq import relaxed_quantize
from training import evaluation_categorical
from training.config_stage2 import load_stage2_config
from training.config_stage3_joint import load_stage3_config


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


def test_canonical_configs_use_4096x16_without_gradient_accumulation() -> None:
    stage2 = load_stage2_config(ROOT / "configs/stage2_entropy.yaml")
    stage3 = load_stage3_config(ROOT / "configs/stage3_joint_vq.yaml")

    assert stage2.model.codebook_size == stage3.model.codebook_size == 4096
    assert stage2.model.codebook_dim == stage3.model.codebook_dim == 16
    assert not hasattr(stage2.train, "gradient_accumulation")
    assert not hasattr(stage3.train, "gradient_accumulation")
    assert rate_parameters(0, stage3) == (0.8, 0.01)
    assert rate_parameters(1000, stage3) == (2.8, 0.01)
    assert rate_parameters(2000, stage3) == (4.8, 0.01)
    assert rate_parameters(10000, stage3) == (4.8, 0.01)
    assert rate_parameters(20000, stage3) == (4.8, 0.01)
    assert rate_parameters(20001, stage3)[1] == 0.1
    assert rate_parameters(21000, stage3) == (8.4, 0.1)
    assert rate_parameters(22000, stage3) == (12.0, 0.1)
    assert rate_parameters(30000, stage3) == (12.0, 0.1)
    assert stage3.loss.gan_start_step == 20000
    assert stage3.loss.gan_weight == 0.0


def test_active_training_scripts_do_not_import_other_scripts() -> None:
    for name in (
        "stage1_tokenizertraining.py",
        "stage2_entropytraining.py",
        "stage3_jointtraining.py",
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
            assert video.shape == (1, 3, 96, 16, 16)
            return torch.zeros(1, 24, 1, 1, 1)

        @staticmethod
        def decode(latent: torch.Tensor) -> torch.Tensor:
            return torch.zeros(1, 3, 96, 16, 16)

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
        assert result["protocol"]["model_input_frames"] == 96
        assert result["protocol"]["bpp_denominator_frames"] == 96
    assert report["aggregate"]["bpp"] == 1.0 / 256.0
    assert raw_report["aggregate"]["bpp"] == 1.0 / 2048.0
