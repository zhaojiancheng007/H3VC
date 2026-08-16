import math

import torch

from entropy_model import CosmosFSQEntropyModel, EntropyModelConfig
from entropy_model.layout import BlockTensor


def tiny_model() -> CosmosFSQEntropyModel:
    torch.manual_seed(11)
    return CosmosFSQEntropyModel(
        EntropyModelConfig(
            window_size=4,
            d_model=32,
            num_layers=2,
            num_heads=4,
            ffn_dim=64,
            max_latent_frames=9,
        )
    ).eval()


def test_entropy_model_shapes_parallel_heads_and_backward() -> None:
    model = tiny_model()
    indices = torch.randint(0, 64000, (2, 2, 8, 8), dtype=torch.int64)
    output = model(indices)

    assert len(output.logits) == 6
    assert [tensor.shape[-1] for tensor in output.logits] == [8, 8, 8, 5, 5, 5]
    assert output.logits[0].shape[:3] == (8, 8, 4)
    assert output.estimated_bits.shape == (2,)
    assert output.bits_per_digit.shape == (6,)
    assert output.bits_per_site.ndim == 0
    assert torch.isfinite(output.bits_per_site)

    model.train()
    model(indices).bits_per_site.backward()
    assert model.output_head.weight.grad is not None
    assert torch.isfinite(model.output_head.weight.grad).all()


def test_zero_logits_match_uniform_fsq_rate() -> None:
    model = tiny_model()
    torch.nn.init.zeros_(model.output_head.weight)
    torch.nn.init.zeros_(model.output_head.bias)
    indices = torch.randint(0, 64000, (1, 2, 8, 8), dtype=torch.int64)
    output = model(indices)
    expected = 3 * math.log2(8) + 3 * math.log2(5)
    assert torch.allclose(output.bits_per_site, torch.tensor(expected), atol=1e-5)


def test_current_and_future_blocks_cannot_change_current_logits() -> None:
    model = tiny_model()
    indices = torch.randint(0, 64000, (1, 2, 8, 8), dtype=torch.int64)
    original = model(indices)

    digits = model.fsq.indices_to_digits(indices)
    digit_blocks = model.layout.to_blocks(digits)
    changed_values = digit_blocks.values.clone()
    current_stage = 3
    for dimension, level in enumerate(model.config.levels):
        changed_values[:, current_stage:, :, dimension] = (
            changed_values[:, current_stage:, :, dimension] + 1
        ) % level
    changed_grid = model.layout.from_blocks(
        BlockTensor(
            changed_values,
            digit_blocks.batch_size,
            digit_blocks.latent_frames,
            digit_blocks.height,
            digit_blocks.width,
            digit_blocks.windows_y,
            digit_blocks.windows_x,
        )
    )
    changed_indices = model.fsq.digits_to_indices(changed_grid)
    changed = model(changed_indices)

    for original_logits, changed_logits in zip(original.logits, changed.logits, strict=True):
        assert torch.equal(
            original_logits[:, : current_stage + 1],
            changed_logits[:, : current_stage + 1],
        )


def test_full_forward_matches_incremental_kv_cache() -> None:
    model = tiny_model()
    indices = torch.randint(0, 64000, (1, 2, 8, 8), dtype=torch.int64)
    codes = model.fsq.indices_to_codes(indices, dtype=model.code_projection.weight.dtype)
    blocks = model.layout.to_blocks(codes).values
    full_logits = model.forward_blocks(blocks)

    cache = None
    for stage in range(blocks.shape[1]):
        previous = None if stage == 0 else blocks[:, stage - 1]
        step_logits, cache = model.forward_step(
            previous,
            stage=stage,
            batch_windows=blocks.shape[0],
            cache=cache,
        )
        for dimension in range(6):
            assert torch.allclose(
                step_logits[dimension],
                full_logits[dimension][:, stage],
                atol=2e-5,
                rtol=1e-5,
            )


def test_incremental_projection_is_stride_invariant() -> None:
    model = tiny_model()
    batch_windows = 4
    sites = model.layout.sites_per_block
    base = torch.randn(batch_windows, sites, model.fsq.dimensions * 2)
    noncontiguous = base[..., ::2]
    assert not noncontiguous.is_contiguous()

    _, cache_a = model.forward_step(
        None, stage=0, batch_windows=batch_windows, cache=None
    )
    _, cache_b = model.forward_step(
        None, stage=0, batch_windows=batch_windows, cache=None
    )
    logits_a, _ = model.forward_step(
        noncontiguous,
        stage=1,
        batch_windows=batch_windows,
        cache=cache_a,
    )
    logits_b, _ = model.forward_step(
        noncontiguous.contiguous(),
        stage=1,
        batch_windows=batch_windows,
        cache=cache_b,
    )

    for left, right in zip(logits_a, logits_b, strict=True):
        assert torch.equal(left, right)
