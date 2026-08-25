import torch

from entropy_model.config_categorical import CategoricalEntropyModelConfig
from entropy_model.layout_dynamic_rank import DynamicRankFourGroupBlockLayout
from entropy_model.model_categorical_dynamic_rank_reliable import (
    CategoricalEntropyModelDynamicRankReliable,
)


def test_dynamic_rank_round_trip_and_expected_w00_schedule() -> None:
    layout = DynamicRankFourGroupBlockLayout(window_size=2)
    grid = torch.arange(1 * 4 * 2 * 4).reshape(1, 4, 2, 4, 1)
    blocks = layout.to_blocks(grid)

    assert blocks.rank_to_group[0, 0, 0].tolist() == [0, 1, 2, 3]
    assert blocks.rank_to_group[0, 0, 1].tolist() == [2, 3, 0, 1]
    assert blocks.rank_to_group[0, 0, 2].tolist() == [1, 0, 3, 2]
    assert blocks.rank_to_group[0, 0, 3].tolist() == [3, 2, 1, 0]
    assert blocks.rank_to_group[0, 1, 0].tolist() == [1, 3, 0, 2]
    torch.testing.assert_close(layout.from_blocks(blocks), grid)


def test_reliability_mask_blocks_generated_cross_frame_only() -> None:
    config = CategoricalEntropyModelConfig(
        codebook_dim=16,
        window_size=2,
        d_model=8,
        num_layers=1,
        num_heads=2,
        ffn_dim=16,
        max_latent_frames=2,
    )
    model = CategoricalEntropyModelDynamicRankReliable(config)
    blocks = model.layout.to_blocks(torch.randn(1, 2, 2, 2, 16))
    status = torch.tensor([[0, 0, 1, 1, 0, 0, 1, 1]])
    mask = model._reliability_attention_mask(blocks, status, torch.float32)[0, 0]
    allowed = mask == 0

    assert allowed[3, 2]
    assert not allowed[4, 3]
    assert allowed[4, 1]
    assert not allowed.triu(diagonal=1).any()


def test_clean_forward_has_expected_outputs() -> None:
    config = CategoricalEntropyModelConfig(
        codebook_dim=16,
        window_size=2,
        d_model=8,
        num_layers=1,
        num_heads=2,
        ffn_dim=16,
        max_latent_frames=2,
    )
    model = CategoricalEntropyModelDynamicRankReliable(config)
    indices = torch.randint(4096, (2, 2, 2, 4))
    codebook = torch.randn(4096, 16)
    quantized = torch.nn.functional.embedding(indices, codebook).permute(0, 4, 1, 2, 3)

    output = model(quantized, indices)

    assert output.estimated_bits.shape == (2,)
    assert torch.isfinite(output.bits_per_index)
