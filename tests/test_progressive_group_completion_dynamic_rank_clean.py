import math

import torch
from torch import nn
import torch.nn.functional as F

from entropy_model.config_categorical import CategoricalEntropyModelConfig
from entropy_model.model_categorical_dynamic_rank_reliable import (
    CategoricalEntropyModelDynamicRankReliable,
)
from training.evaluation_progressive_groups_dynamic_rank_clean import (
    complete_progressive_groups_dynamic_rank_clean,
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


class RecordingEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding) -> None:
        super().__init__()
        self.embedding = embedding
        self.inputs: list[torch.Tensor] = []

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        self.inputs.append(values.detach().clone())
        return self.embedding(values)


def test_level_two_keeps_half_of_each_dynamic_rank_schedule() -> None:
    torch.manual_seed(5)
    model = build_model()
    recorder = RecordingEmbedding(model.source_status_embedding)
    model.source_status_embedding = recorder
    codebook = torch.randn(4096, 16)
    indices = torch.randint(4096, (1, 2, 2, 4))

    result = complete_progressive_groups_dynamic_rank_clean(
        model, codebook, indices, retained_groups=2
    )

    true_blocks = model.layout.to_blocks(indices.unsqueeze(-1)).values.squeeze(-1)
    result_blocks = model.layout.to_blocks(result.indices.unsqueeze(-1)).values.squeeze(-1)
    torch.testing.assert_close(result_blocks[:, 0::4], true_blocks[:, 0::4])
    torch.testing.assert_close(result_blocks[:, 1::4], true_blocks[:, 1::4])
    assert all(not status.any() for status in recorder.inputs)
    assert result.retained_indices == indices.numel() // 2
    assert result.generated_indices == indices.numel() // 2


def test_level_four_matches_clean_teacher_forcing() -> None:
    torch.manual_seed(7)
    model = build_model()
    codebook = torch.randn(4096, 16)
    indices = torch.randint(4096, (1, 2, 2, 4))

    result = complete_progressive_groups_dynamic_rank_clean(
        model, codebook, indices, retained_groups=4
    )

    quantized = F.embedding(indices, codebook).permute(0, 4, 1, 2, 3)
    expected = model(quantized, indices)
    torch.testing.assert_close(result.indices, indices)
    torch.testing.assert_close(result.retained_bits, expected.estimated_bits.sum())
    assert result.generated_indices == 0
