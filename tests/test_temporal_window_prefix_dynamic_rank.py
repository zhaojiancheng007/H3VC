import math

import torch
import torch.nn.functional as F

from entropy_model.config_categorical import CategoricalEntropyModelConfig
from entropy_model.model_categorical_dynamic_rank_reliable import (
    CategoricalEntropyModelDynamicRankReliable,
)
from training.evaluation_temporal_window_prefix_dynamic_rank import (
    complete_temporal_window_prefix_dynamic_rank,
    decode_window_mask,
    encode_window_mask,
    fixed_cardinality_mask_bits,
    select_dynamic_windows,
    temporal_window_change_scores,
)


def build_model() -> CategoricalEntropyModelDynamicRankReliable:
    model = CategoricalEntropyModelDynamicRankReliable(
        CategoricalEntropyModelConfig(
            codebook_size=4096,
            codebook_dim=16,
            window_size=2,
            d_model=8,
            num_layers=1,
            num_heads=2,
            ffn_dim=16,
            max_latent_frames=2,
        )
    ).eval()
    torch.nn.init.normal_(model.output_head.weight)
    return model


def test_fixed_cardinality_window_mask_round_trip_and_bits() -> None:
    mask = torch.tensor([True, False, True, False])
    rank = encode_window_mask(mask)
    decoded = decode_window_mask(rank, num_windows=4, retained_windows=2)

    torch.testing.assert_close(decoded, mask)
    assert fixed_cardinality_mask_bits(4, 2) == 3
    assert fixed_cardinality_mask_bits(4, 4) == 0


def test_change_score_selects_the_window_with_largest_temporal_innovation() -> None:
    codebook = torch.arange(16, dtype=torch.float32).unsqueeze(1).repeat(1, 4)
    reference = torch.zeros((1, 4, 4), dtype=torch.int64)
    current = reference.clone()
    current[:, :2, 2:] = 3
    current[:, 2:, :2] = 7

    scores = temporal_window_change_scores(
        current, reference, codebook, window_size=2
    )
    selected = select_dynamic_windows(scores, retained_windows=1)

    assert scores.tolist() == [[0.0, 9.0, 49.0, 0.0]]
    assert selected.tolist() == [[False, False, True, False]]


def test_all_windows_matches_clean_teacher_forcing_and_counts_no_side_bits() -> None:
    torch.manual_seed(17)
    model = build_model()
    codebook = torch.randn(4096, 16)
    indices = torch.randint(4096, (1, 2, 4, 4))

    result = complete_temporal_window_prefix_dynamic_rank(
        model, codebook, indices, retained_windows=4
    )
    quantized = F.embedding(indices, codebook).permute(0, 4, 1, 2, 3)
    expected = model(quantized, indices)

    torch.testing.assert_close(result.indices, indices)
    torch.testing.assert_close(result.symbol_bits, expected.estimated_bits.sum())
    assert result.side_bits == 0
    assert result.generated_indices == 0
    assert result.retained_indices == indices.numel()


def test_one_dynamic_window_transmits_anchor_plus_selected_refresh() -> None:
    torch.manual_seed(23)
    model = build_model()
    codebook = torch.randn(4096, 16)
    indices = torch.randint(4096, (1, 2, 4, 4))

    result = complete_temporal_window_prefix_dynamic_rank(
        model, codebook, indices, retained_windows=1
    )

    sites_per_window = 4
    assert result.retained_indices == (4 + 1) * sites_per_window
    assert result.generated_indices == 3 * sites_per_window
    assert result.retained_windows == 5
    assert result.total_windows == 8
    assert result.side_bits == math.ceil(math.log2(math.comb(4, 1)))
