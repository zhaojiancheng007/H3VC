from pathlib import Path

import pytest
import torch

from entropy_model import CosmosFSQEntropyModel, EntropyModelConfig
from entropy_model.checkpoint import (
    checkpoint_sha256,
    load_entropy_checkpoint,
    save_entropy_checkpoint,
)
from training.config_ar import load_ar_training_config


def test_entropy_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(23)
    config = EntropyModelConfig(
        window_size=4,
        d_model=32,
        num_layers=1,
        num_heads=4,
        ffn_dim=64,
        max_latent_frames=2,
    )
    model = CosmosFSQEntropyModel(config)
    path = tmp_path / "model.pt"

    save_entropy_checkpoint(path, model, step=7)
    restored, payload = load_entropy_checkpoint(path)

    assert restored.config == config
    assert payload["step"] == 7
    assert len(checkpoint_sha256(path)) == 64
    for name, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)


def test_ar_yaml_config_is_strict(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
model:
  window_size: 4
  d_model: 32
  num_layers: 1
  num_heads: 4
  ffn_dim: 64
  max_latent_frames: 9
optimizer:
  learning_rate: 0.0001
train:
  batch_size: 1
  gradient_accumulation: 1
  max_steps: 1
  num_workers: 0
  log_interval: 1
  checkpoint_interval: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_ar_training_config(path)
    assert config.model.d_model == 32
    assert config.optimizer.learning_rate == pytest.approx(0.0001)
    assert config.train.max_steps == 1

    path.write_text(path.read_text(encoding="utf-8") + "  unknown_field: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown TrainLoopConfig fields"):
        load_ar_training_config(path)
