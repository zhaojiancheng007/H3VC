import json
from pathlib import Path

import torch
import torch.nn as nn

from scripts.stage1 import (
    CHECKPOINT_INTERVAL,
    LPIPS_WEIGHT,
    MSE_WEIGHT,
    keep_best_lpips_checkpoint,
    save_checkpoint,
)
from tokenizer.stage1_vq import (
    QUANTIZER_GEOMETRY,
    EMAVectorQuantizer,
    Stage1VQTokenizer,
)
from training.config_stage1 import (
    learning_rate_at_step,
    load_stage1_training_config,
)


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv3d(3, 16, kernel_size=1)
        self.decoder = nn.Conv3d(16, 3, kernel_size=1)

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        return self.encoder(video)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)


def test_stage1_ema_vq_forward_and_gradient_path() -> None:
    torch.manual_seed(3)
    model = Stage1VQTokenizer(TinyBackbone(), codebook_size=4, codebook_dim=16)
    video = torch.randn(1, 3, 1, 2, 2)

    output = model(video)
    loss = output.reconstruction.square().mean() + output.commitment_loss
    loss.backward()

    assert output.indices.shape == (1, 1, 2, 2)
    assert model.quantizer.initted.item()
    assert model.backbone.encoder.weight.grad is not None
    assert model.backbone.decoder.weight.grad is not None
    assert model.pre_quant.weight.grad is not None
    assert model.post_quant.weight.grad is not None
    assert list(model.quantizer.parameters()) == []


def test_identity_initialized_projections_preserve_raw_latent() -> None:
    model = Stage1VQTokenizer(TinyBackbone(), codebook_size=4, codebook_dim=16)
    latent = torch.randn(1, 16, 2, 3, 4)

    projected = model.pre_quant(latent)
    reconstructed_latent = model.post_quant(projected)

    assert QUANTIZER_GEOMETRY == "raw_l2"
    torch.testing.assert_close(projected, latent, rtol=0, atol=0)
    torch.testing.assert_close(reconstructed_latent, latent, rtol=0, atol=0)


def test_eval_does_not_update_ema_codebook() -> None:
    quantizer = EMAVectorQuantizer(codebook_size=4, codebook_dim=16)
    projected = torch.randn(1, 16, 1, 2, 2)
    quantizer(projected)
    before = {name: value.clone() for name, value in quantizer.state_dict().items()}

    quantizer.eval()
    quantizer(projected * 2.0)

    for name, value in quantizer.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_stage1_replaces_codes_below_cluster_size_one() -> None:
    quantizer = EMAVectorQuantizer(
        codebook_size=4,
        codebook_dim=2,
        samples_per_rank=2,
    )
    quantizer.initted.fill_(True)
    quantizer.embed.copy_(
        torch.tensor(
            [
                [0.0, 0.0],
                [10.0, 10.0],
                [20.0, 20.0],
                [30.0, 30.0],
            ]
        )
    )
    quantizer.cluster_size.copy_(torch.tensor([10.0, 0.5, 0.1, 2.0]))
    quantizer.embed_sum.copy_(
        quantizer.embed * quantizer.cluster_size[:, None]
    )

    projected = torch.zeros(1, 2, 1, 1, 1)
    quantizer(projected)

    torch.testing.assert_close(
        quantizer.cluster_size,
        torch.tensor([9.91, 1.0, 1.0, 1.98]),
    )
    assert quantizer.num_updates.item() == 1


def test_stage1_loads_checkpoint_without_revival_update_counter() -> None:
    quantizer = EMAVectorQuantizer(codebook_size=4, codebook_dim=2)
    old_state = quantizer.state_dict()
    old_state.pop("num_updates")

    quantizer.load_state_dict(old_state, strict=True)

    assert quantizer.num_updates.item() == 0


def test_checkpoint_contains_general_stage1_state(tmp_path) -> None:
    model = Stage1VQTokenizer(TinyBackbone(), codebook_size=4, codebook_dim=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    path = tmp_path / "step_0005000.pt"

    save_checkpoint(
        path,
        model,
        optimizer,
        "wanvae",
        input_frames=97,
        latent_channels=16,
        codebook_dim=16,
        config={"train": {}},
        step=5000,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert CHECKPOINT_INTERVAL == 5000
    assert MSE_WEIGHT == LPIPS_WEIGHT == 1.0
    assert payload["tokenizer"] == "wanvae"
    assert payload["step"] == 5000
    assert payload["quantizer_geometry"] == "raw_l2"
    assert set(payload["model"]) == set(model.state_dict())


def test_h3_107_frame_piecewise_schedule_and_best_policy() -> None:
    config = load_stage1_training_config(
        Path("configs/stage1.yaml")
    )

    assert config.train.max_steps == 60_000
    assert config.train.checkpoint_interval == 5_000
    assert config.train.checkpoint_policy == "best_lpips"
    assert learning_rate_at_step(config.optimizer, 0) == 1e-4
    assert learning_rate_at_step(config.optimizer, 14_999) == 1e-4
    assert learning_rate_at_step(config.optimizer, 15_000) == 5e-5
    assert learning_rate_at_step(config.optimizer, 29_999) == 5e-5
    assert learning_rate_at_step(config.optimizer, 30_000) == 1e-5
    assert learning_rate_at_step(config.optimizer, 44_999) == 1e-5
    assert learning_rate_at_step(config.optimizer, 45_000) == 5e-6
    assert learning_rate_at_step(config.optimizer, 60_000) == 5e-6


def test_best_lpips_policy_keeps_only_the_best_checkpoint(tmp_path: Path) -> None:
    candidate = tmp_path / ".candidate.pt"
    best = tmp_path / "checkpoint_best.pt"
    record = tmp_path / "best_checkpoint.json"
    evaluation = tmp_path / "evaluation.json"

    def write_candidate(step: int, lpips: float) -> None:
        candidate.write_bytes(f"checkpoint-{step}".encode())
        evaluation.write_text(
            json.dumps(
                {
                    "checkpoint_step": step,
                    "aggregate": {
                        "psnr": 30.0,
                        "bpp": 0.01,
                        "lpips": lpips,
                        "dists": 0.05,
                        "latency_ms": 1.0,
                    },
                }
            ),
            encoding="utf-8",
        )

    write_candidate(5_000, 0.2)
    assert keep_best_lpips_checkpoint(candidate, evaluation, best, record)
    assert best.read_bytes() == b"checkpoint-5000"

    write_candidate(10_000, 0.3)
    assert not keep_best_lpips_checkpoint(candidate, evaluation, best, record)
    assert not candidate.exists()
    assert best.read_bytes() == b"checkpoint-5000"

    write_candidate(15_000, 0.1)
