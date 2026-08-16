
import torch

from entropy_model.config_wan_vq import WanVQEntropyModelConfig
from entropy_model.model_wan_vq_categorical import (
    WanVQCategoricalEntropyModel,
)


def make_model() -> WanVQCategoricalEntropyModel:
    config = WanVQEntropyModelConfig(
        window_size=4,
        d_model=32,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        max_latent_frames=2,
    )
    return WanVQCategoricalEntropyModel(config)


def test_predicts_one_4096_way_distribution_per_index() -> None:
    model = make_model()
    features = torch.randn(1, 32, 2, 4, 4)
    indices = torch.randint(0, 4096, (1, 2, 4, 4))

    output = model(features, indices)
    assert output.logits.shape == (1, 8, 4, 4096)
    assert torch.allclose(output.bits_per_site, torch.tensor(12.0))
    output.bits_per_site.backward()
    assert model.output_head.weight.grad is not None


def test_current_group_features_cannot_change_current_or_past_logits() -> None:
    model = make_model().eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.output_head.weight)
    features = torch.randn(1, 32, 2, 4, 4)
    original, _ = model.predict_quant_features(features)

    changed = features.clone()
    changed[:, :, 0, 0::2, 1::2] += 10.0
    modified, _ = model.predict_quant_features(changed)

    current_group = 1
    assert torch.equal(
        original[:, : current_group + 1],
        modified[:, : current_group + 1],
    )
    assert not torch.equal(original[:, current_group + 1 :], modified[:, current_group + 1 :])
