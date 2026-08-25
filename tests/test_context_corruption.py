import torch
import torch.nn.functional as F

from entropy_model.config_categorical import CategoricalEntropyModelConfig
from entropy_model.model_categorical import CategoricalEntropyOutput
from entropy_model.model_categorical_context_corruption import (
    CategoricalEntropyModelContextCorruption,
    ContextCorruptionEntropyOutput,
    build_codebook_knn,
)


def _model() -> CategoricalEntropyModelContextCorruption:
    config = CategoricalEntropyModelConfig(
        codebook_size=4096,
        codebook_dim=16,
        window_size=2,
        d_model=8,
        num_layers=1,
        num_heads=2,
        ffn_dim=16,
        max_latent_frames=2,
    )
    model = CategoricalEntropyModelContextCorruption(config)
    torch.nn.init.normal_(model.output_head.weight)
    return model


def test_codebook_knn_excludes_the_source_index() -> None:
    codebook = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    neighbors = build_codebook_knn(codebook, 2)

    assert neighbors.shape == (8, 2)
    assert not torch.any(neighbors == torch.arange(8)[:, None])
    assert neighbors[0, 0].item() == 1


def test_single_forward_combines_clean_and_corrupt_losses() -> None:
    torch.manual_seed(7)
    model = _model()
    codebook = torch.randn(4096, 16)
    indices = torch.randint(0, 4096, (4, 1, 2, 2))
    quantized = F.embedding(indices, codebook).permute(0, 4, 1, 2, 3)
    corrupted = torch.tensor([False, True, False, False])
    retained = torch.tensor([4, 2, 4, 4])

    output = model(
        quantized,
        indices,
        codebook=codebook,
        knn_indices=((torch.arange(4096) + 1) % 4096)[:, None],
        corrupted_samples=corrupted,
        retained_groups=retained,
        corrupt_loss_weight=0.25,
    )

    assert isinstance(output, ContextCorruptionEntropyOutput)
    assert output.estimated_bits.shape == (4,)
    torch.testing.assert_close(
        output.loss,
        output.clean_bits_per_index + 0.25 * output.corrupt_bits_per_index,
    )
    output.loss.backward()
    status_grad = model.source_status_embedding.weight.grad
    assert status_grad is not None
    assert status_grad[0].abs().sum() > 0
    assert status_grad[1].abs().sum() > 0


def test_clean_evaluation_path_uses_the_parallel_model() -> None:
    model = _model()
    codebook = torch.randn(4096, 16)
    indices = torch.randint(0, 4096, (1, 1, 2, 2))
    quantized = F.embedding(indices, codebook).permute(0, 4, 1, 2, 3)

    output = model(quantized, indices)

    assert isinstance(output, CategoricalEntropyOutput)
    assert output.estimated_bits.shape == (1,)
