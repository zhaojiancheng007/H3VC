import torch

from entropy_model.config_categorical import CategoricalEntropyModelConfig
from entropy_model.model_categorical import CategoricalEntropyModel
from tokenizer.stage3_vq import selected_code_vectors
from training.evaluation_progressive_groups_distributed import (
    complete_progressive_groups,
)


def build_model() -> CategoricalEntropyModel:
    return CategoricalEntropyModel(
        CategoricalEntropyModelConfig(
            codebook_size=4096,
            codebook_dim=24,
            window_size=8,
            d_model=24,
            num_layers=1,
            num_heads=1,
            ffn_dim=32,
            max_latent_frames=2,
        )
    ).eval()


def test_level_four_preserves_every_index() -> None:
    torch.manual_seed(7)
    model = build_model()
    codebook = torch.randn(4096, 24)
    indices = torch.randint(4096, (1, 2, 8, 8))

    result = complete_progressive_groups(
        model, codebook, indices, retained_groups=4
    )

    torch.testing.assert_close(result.indices, indices)
    full = model(selected_code_vectors(codebook, indices), indices)
    torch.testing.assert_close(result.retained_bits, full.estimated_bits.sum())
    assert result.retained_indices == indices.numel()
    assert result.generated_indices == 0
    assert torch.isfinite(result.retained_bits)


def test_level_two_keeps_first_two_groups_and_generates_the_rest() -> None:
    torch.manual_seed(11)
    model = build_model()
    codebook = torch.randn(4096, 24)
    indices = torch.randint(4096, (1, 1, 8, 8))

    result = complete_progressive_groups(
        model, codebook, indices, retained_groups=2
    )

    for group, positions in enumerate(model.layout.positions):
        y = torch.tensor(positions) // 8
        x = torch.tensor(positions) % 8
        if group < 2:
            torch.testing.assert_close(
                result.indices[0, 0, y, x], indices[0, 0, y, x]
            )
    assert result.retained_indices == indices.numel() // 2
    assert result.generated_indices == indices.numel() // 2
    assert result.indices.amin() >= 0
    assert result.indices.amax() < 4096
