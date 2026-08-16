import math

import torch

from entropy_model.config_row_group import RowGroupEntropyModelConfig
from entropy_model.layout_row_group import RowGroupLayout
from entropy_model.model_quant_feature_row_group import CosmosFSQRowGroupEntropyModel


def tiny_model() -> CosmosFSQRowGroupEntropyModel:
    torch.manual_seed(41)
    return CosmosFSQRowGroupEntropyModel(
        RowGroupEntropyModelConfig(
            rows_per_group=4,
            d_model=32,
            num_layers=2,
            num_heads=4,
            ffn_dim=64,
            max_latent_frames=3,
            max_latent_height=8,
            max_latent_width=6,
        )
    )


def token_pair(
    model: CosmosFSQRowGroupEntropyModel,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randint(0, 64000, (2, 3, 8, 6), dtype=torch.int64)
    features = model.fsq.indices_to_codes(indices).permute(0, 4, 1, 2, 3)
    return features, indices


def test_row_group_layout_round_trip() -> None:
    layout = RowGroupLayout(rows_per_group=4)
    grid = torch.arange(2 * 3 * 8 * 6 * 2).reshape(2, 3, 8, 6, 2)
    blocks = layout.to_blocks(grid)

    assert blocks.values.shape == (2, 6, 24, 2)
    assert torch.equal(layout.from_blocks(blocks), grid)


def test_forward_outputs_six_parallel_digit_distributions_and_exact_rate() -> None:
    model = tiny_model()
    features, indices = token_pair(model)
    output = model(features, indices)
    targets = model.layout.to_blocks(model.fsq.indices_to_digits(indices)).values

    assert [logits.shape for logits in output.logits] == [
        (2, 6, 24, level) for level in model.config.levels
    ]
    manual_bits = []
    for dimension, logits in enumerate(output.logits):
        log2_prob = torch.log_softmax(logits.float(), dim=-1) / math.log(2.0)
        manual_bits.append(
            -log2_prob.gather(-1, targets[..., dimension, None]).squeeze(-1)
        )
    expected = torch.stack(manual_bits, dim=-1).sum(dim=(1, 2, 3))
    assert torch.allclose(output.estimated_bits, expected, atol=1e-5)
    assert torch.allclose(output.bits_per_site, expected.sum() / indices.numel())


def test_parallel_teacher_forcing_matches_incremental_logits() -> None:
    model = tiny_model().eval()
    features, _ = token_pair(model)
    parallel = model.predict_quant_features(features)
    blocks = parallel.code_blocks.values
    cache = None
    stage_logits = []
    for stage in range(blocks.shape[1]):
        previous = None if stage == 0 else blocks[:, stage - 1]
        logits, cache = model.forward_step(
            previous,
            stage=stage,
            batch_size=blocks.shape[0],
            latent_frames=parallel.code_blocks.latent_frames,
            latent_height=parallel.code_blocks.height,
            latent_width=parallel.code_blocks.width,
            cache=cache,
        )
        stage_logits.append(logits)

    for dimension, parallel_logits in enumerate(parallel.logits):
        serial_logits = torch.stack(
            [logits[dimension] for logits in stage_logits], dim=1
        )
        assert torch.allclose(parallel_logits, serial_logits, atol=1e-6, rtol=1e-6)


def test_current_group_codes_cannot_change_current_or_past_logits() -> None:
    model = tiny_model().eval()
    features, _ = token_pair(model)
    original = model.predict_quant_features(features).logits
    blocks = model.layout.to_blocks(features.permute(0, 2, 3, 4, 1))
    changed = blocks.values.clone()
    changed[:, 3] = -changed[:, 3]
    changed_features = model.layout.from_blocks(
        type(blocks)(
            changed,
            blocks.batch_size,
            blocks.latent_frames,
            blocks.height,
            blocks.width,
            blocks.rows_per_group,
        )
    ).permute(0, 4, 1, 2, 3)
    modified = model.predict_quant_features(changed_features).logits

    for left, right in zip(original, modified, strict=True):
        assert torch.equal(left[:, :4], right[:, :4])
    assert any(
        not torch.equal(left[:, 4], right[:, 4])
        for left, right in zip(original, modified, strict=True)
    )


def test_cache_keeps_only_previous_frame_and_decoded_current_groups() -> None:
    model = tiny_model().eval()
    features, _ = token_pair(model)
    blocks = model.layout.to_blocks(features.permute(0, 2, 3, 4, 1))
    cache = None
    cached_stages = []
    for stage in range(blocks.values.shape[1]):
        previous = None if stage == 0 else blocks.values[:, stage - 1]
        _, cache = model.forward_step(
            previous,
            stage=stage,
            batch_size=2,
            latent_frames=3,
            latent_height=8,
            latent_width=6,
            cache=cache,
        )
        cached_stages.append(model._cache_length(cache) // 24)

    assert cached_stages == [1, 2, 2, 3, 2, 3]


def test_rate_backward_reaches_model_and_quant_context() -> None:
    model = tiny_model()
    features, indices = token_pair(model)
    features.requires_grad_()
    model(features, indices).bits_per_site.backward()

    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is None
        or not torch.isfinite(parameter.grad).all()
        or parameter.grad.abs().sum().item() == 0
    ]
    assert missing == []
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    gradient_blocks = model.layout.to_blocks(
        features.grad.permute(0, 2, 3, 4, 1)
    ).values
    assert gradient_blocks[:, :-1].abs().sum().item() > 0
    assert torch.count_nonzero(gradient_blocks[:, -1]).item() == 0
