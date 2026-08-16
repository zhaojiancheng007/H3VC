from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer.wan_vq.model_standard import WanVQStandardAdapter
from training.config_stage1_wan_vq_standard import load_wan_vq_training_config


def make_quantizer() -> WanVQStandardAdapter:
    model = WanVQStandardAdapter.__new__(WanVQStandardAdapter)
    nn.Module.__init__(model)
    model.codebook_size = 4
    model.codebook_dim = 2
    model.codebook = nn.Embedding(4, 2)
    with torch.no_grad():
        model.codebook.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0], [0.0, -4.0]])
        )
    return model


def test_normalized_vq_lookup_and_gradient_routing() -> None:
    model = make_quantizer()
    projected = torch.tensor(
        [[[[[2.0, 1.0]]], [[[1.0, -3.0]]]]], requires_grad=True
    )

    quantized, normalized, selected, indices = model.quantize(projected)

    assert indices.tolist() == [[[[0, 3]]]]
    assert torch.allclose(torch.linalg.vector_norm(normalized, dim=1), torch.ones(1, 1, 1, 2))
    assert torch.allclose(torch.linalg.vector_norm(selected, dim=1), torch.ones(1, 1, 1, 2))
    assert torch.equal(quantized, selected)

    codebook_loss = F.mse_loss(selected, normalized.detach())
    commitment_loss = 0.25 * F.mse_loss(normalized, selected.detach())
    (codebook_loss + commitment_loss).backward()

    assert projected.grad is not None
    assert torch.count_nonzero(projected.grad) > 0
    assert model.codebook.weight.grad is not None
    assert torch.count_nonzero(model.codebook.weight.grad) > 0


def test_standard_training_config() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "stage1_wan_vq_4096x32_standard_30k.yaml"
    )
    config = load_wan_vq_training_config(config_path)

    assert config.optimizer.learning_rate == 1e-4
    assert config.optimizer.betas == (0.9, 0.99)
    assert config.loss.commitment_weight == 0.25
    assert config.train.max_steps == 30_000
