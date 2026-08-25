import torch

from entropy_model.config_categorical import CategoricalEntropyModelConfig
from entropy_model.model_categorical_context_corruption import (
    CategoricalEntropyModelContextCorruption,
)
from training.temporal_attention_mask_ablation import logits_with_temporal_radius


def test_unrestricted_diagnostic_forward_matches_original_model() -> None:
    torch.manual_seed(71)
    config = CategoricalEntropyModelConfig(
        codebook_dim=16,
        window_size=8,
        d_model=8,
        num_layers=1,
        num_heads=2,
        ffn_dim=16,
        max_latent_frames=3,
    )
    model = CategoricalEntropyModelContextCorruption(config).eval()
    selected = torch.randn(1, 16, 3, 8, 8)

    expected, _ = model.predict(selected)
    actual, _ = logits_with_temporal_radius(model, selected, radius=None)

    assert torch.equal(actual, expected)
