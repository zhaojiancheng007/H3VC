import math

import torch

from entropy_model.transformer import (
    BlockCausalSelfAttention,
    BlockCausalTransformer,
)


def test_block_causal_attention_mask_has_exact_stage_dependencies() -> None:
    mask = BlockCausalTransformer.attention_mask(
        num_blocks=3,
        sites_per_block=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    block_ids = torch.arange(3).repeat_interleave(2)
    expected_allowed = block_ids[:, None] >= block_ids[None, :]

    assert mask.shape == (1, 1, 6, 6)
    assert torch.equal(mask[0, 0] == 0, expected_allowed)
    assert torch.equal(torch.isneginf(mask[0, 0]), ~expected_allowed)


def test_scaled_dot_product_attention_matches_explicit_reference() -> None:
    torch.manual_seed(47)
    attention = BlockCausalSelfAttention(d_model=8, num_heads=2).eval()
    values = torch.randn(2, 6, 8)
    mask = BlockCausalTransformer.attention_mask(
        num_blocks=3,
        sites_per_block=2,
        device=values.device,
        dtype=values.dtype,
    )

    query, key, value = attention._qkv(values)
    scores = query @ key.transpose(-2, -1) / math.sqrt(attention.head_dim)
    probabilities = torch.softmax(scores + mask, dim=-1)
    mixed = probabilities @ value
    expected = attention.output(mixed.transpose(1, 2).reshape(values.shape))
    actual = attention(values, mask)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_future_stage_values_cannot_change_past_attention_outputs() -> None:
    torch.manual_seed(53)
    attention = BlockCausalSelfAttention(d_model=8, num_heads=2).eval()
    values = torch.randn(1, 6, 8)
    changed = values.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 100.0
    mask = BlockCausalTransformer.attention_mask(
        num_blocks=3,
        sites_per_block=2,
        device=values.device,
        dtype=values.dtype,
    )

    original_output = attention(values, mask)
    changed_output = attention(changed, mask)

    assert torch.equal(original_output[:, :4], changed_output[:, :4])
