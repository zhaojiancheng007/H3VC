import math

import torch
from torch import nn
import torch.nn.functional as F

from entropy_model.config_categorical import CategoricalEntropyModelConfig
from entropy_model.model_categorical_context_corruption import (
    CategoricalEntropyModelContextCorruption,
)
from tokenizer.stage3_vq import selected_code_vectors
from training.evaluation_progressive_groups_context_corruption import (
    complete_progressive_groups_context_corruption,
)


class RecordingEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding) -> None:
        super().__init__()
        self.embedding = embedding
        self.inputs: list[torch.Tensor] = []

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        self.inputs.append(values.detach().clone())
        return self.embedding(values)


def build_model() -> CategoricalEntropyModelContextCorruption:
    model = CategoricalEntropyModelContextCorruption(
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
    torch.nn.init.normal_(model.source_status_embedding.weight)
    return model


def test_level_two_marks_generated_context_for_the_next_block() -> None:
    torch.manual_seed(5)
    model = build_model()
    recorder = RecordingEmbedding(model.source_status_embedding)
    model.source_status_embedding = recorder
    codebook = torch.randn(4096, 16)
    indices = torch.randint(4096, (1, 1, 2, 2))

    result = complete_progressive_groups_context_corruption(
        model, codebook, indices, retained_groups=2
    )

    assert [int(status[0, 0]) for status in recorder.inputs] == [0, 0, 1]
    assert result.retained_indices == indices.numel() // 2
    assert result.generated_indices == indices.numel() // 2


def test_level_four_matches_status_aware_teacher_forcing() -> None:
    torch.manual_seed(7)
    model = build_model()
    codebook = torch.randn(4096, 16)
    indices = torch.randint(4096, (1, 2, 2, 2))

    result = complete_progressive_groups_context_corruption(
        model, codebook, indices, retained_groups=4
    )

    selected = selected_code_vectors(codebook, indices)
    blocks = model.layout.to_blocks(selected.permute(0, 2, 3, 4, 1))
    source_status = torch.zeros(
        blocks.values.shape[:2], dtype=torch.long
    )
    hidden = model.transformer(
        model._shifted_inputs(blocks.values, source_status),
        model.layout.sites_per_block,
    )
    logits = model.output_head(model.output_norm(hidden))
    targets = model._targets(indices)
    expected_bits = F.cross_entropy(
        logits.float().reshape(-1, model.config.codebook_size),
        targets.reshape(-1),
        reduction="sum",
    ) / math.log(2.0)

    torch.testing.assert_close(result.indices, indices)
    torch.testing.assert_close(result.retained_bits, expected_bits)
    assert result.retained_indices == indices.numel()
    assert result.generated_indices == 0
