import math

import pytest
import torch

from entropy_model import EntropyModelConfig
from entropy_model.layout import BlockTensor
from entropy_model.model import CosmosFSQEntropyModel
from entropy_model.model_quant_feature import CosmosFSQQuantFeatureEntropyModel


def tiny_model() -> CosmosFSQQuantFeatureEntropyModel:
    torch.manual_seed(31)
    return CosmosFSQQuantFeatureEntropyModel(
        EntropyModelConfig(
            window_size=4,
            d_model=32,
            num_layers=2,
            num_heads=4,
            ffn_dim=64,
            max_latent_frames=3,
        )
    )


def token_pair(
    model: CosmosFSQQuantFeatureEntropyModel,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randint(0, 64000, (2, 2, 8, 8), dtype=torch.int64)
    features = model.fsq.indices_to_codes(indices).permute(0, 4, 1, 2, 3)
    return features, indices


def randomize_output_head(model: CosmosFSQQuantFeatureEntropyModel) -> None:
    torch.nn.init.normal_(model.output_head.weight, std=0.02)
    torch.nn.init.normal_(model.output_head.bias, std=0.02)


def test_quant_feature_forward_has_six_parallel_categorical_heads() -> None:
    model = tiny_model()
    features, indices = token_pair(model)
    output = model(features, indices)

    assert torch.isfinite(output.bits_per_site)
    assert len(output.logits) == 6
    assert [logits.shape[-1] for logits in output.logits] == [8, 8, 8, 5, 5, 5]
    assert any(torch.count_nonzero(logits).item() > 0 for logits in output.logits)


def test_quant_feature_rate_matches_manual_negative_log2_probability() -> None:
    model = tiny_model()
    randomize_output_head(model)
    features, indices = token_pair(model)
    output = model(features, indices)
    targets = model.layout.to_blocks(model.fsq.indices_to_digits(indices)).values

    manual_bits = []
    for dimension, logits in enumerate(output.logits):
        log2_probabilities = torch.log_softmax(logits.float(), dim=-1) / math.log(2.0)
        target = targets[..., dimension].unsqueeze(-1)
        manual_bits.append(-log2_probabilities.gather(-1, target).squeeze(-1))
    manual = torch.stack(manual_bits, dim=-1)
    expected_per_window = manual.sum(dim=(-1, -2, -3))
    expected_per_clip = expected_per_window.reshape(
        indices.shape[0], -1
    ).sum(dim=1)

    assert torch.allclose(output.estimated_bits, expected_per_clip, atol=1e-5)
    assert torch.allclose(
        output.bits_per_site,
        expected_per_clip.sum() / indices.numel(),
        atol=1e-6,
    )


def test_quant_feature_forward_rejects_feature_target_mismatch() -> None:
    model = tiny_model()
    features, indices = token_pair(model)
    changed = features.clone()
    changed[0, 0, 0, 0, 0] += 0.25

    with pytest.raises(ValueError, match="do not represent"):
        model(changed, indices)


def test_current_quant_group_cannot_change_current_or_past_logits() -> None:
    model = tiny_model()
    randomize_output_head(model)
    features, _ = token_pair(model)
    original = model.predict_quant_features(features)

    grid = features.permute(0, 2, 3, 4, 1)
    blocks = model.layout.to_blocks(grid)
    changed_values = blocks.values.clone()
    current_stage = 3
    changed_values[:, current_stage] = -changed_values[:, current_stage]
    changed_grid = model.layout.from_blocks(
        BlockTensor(
            changed_values,
            blocks.batch_size,
            blocks.latent_frames,
            blocks.height,
            blocks.width,
            blocks.windows_y,
            blocks.windows_x,
        )
    )
    changed_features = changed_grid.permute(0, 4, 1, 2, 3)
    changed = model.predict_quant_features(changed_features)

    for original_logits, changed_logits in zip(
        original.logits, changed.logits, strict=True
    ):
        assert torch.equal(
            original_logits[:, : current_stage + 1],
            changed_logits[:, : current_stage + 1],
        )
    assert any(
        not torch.equal(left[:, current_stage + 1], right[:, current_stage + 1])
        for left, right in zip(original.logits, changed.logits, strict=True)
    )


def test_rate_loss_reaches_every_trainable_parameter_and_quant_context() -> None:
    model = tiny_model()
    features, indices = token_pair(model)
    features.requires_grad_()

    model(features, indices).bits_per_site.backward()

    missing = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            missing.append(name)
        elif parameter.grad.abs().sum().item() == 0:
            missing.append(name)
    assert missing == []
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum().item() > 0
    gradient_blocks = model.layout.to_blocks(
        features.grad.permute(0, 2, 3, 4, 1)
    ).values
    assert gradient_blocks[:, :-1].abs().sum().item() > 0
    assert torch.count_nonzero(gradient_blocks[:, -1]).item() == 0


def test_quant_feature_forward_matches_index_reference_model() -> None:
    model = tiny_model().eval()
    randomize_output_head(model)
    reference = CosmosFSQEntropyModel(model.config).eval()
    reference.load_state_dict(model.state_dict(), strict=True)
    features, indices = token_pair(model)

    quant_output = model(features, indices)
    index_output = reference(indices)

    for quant_logits, index_logits in zip(
        quant_output.logits, index_output.logits, strict=True
    ):
        assert torch.equal(quant_logits, index_logits)
    assert torch.equal(quant_output.estimated_bits, index_output.estimated_bits)
    assert torch.equal(quant_output.bits_per_site, index_output.bits_per_site)
    assert torch.equal(quant_output.bits_per_digit, index_output.bits_per_digit)
