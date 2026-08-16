import torch
import torch.nn.functional as F

from tokenizer.wan_vq.model_stage3 import quantize_with_relaxed_probabilities


def test_shared_distance_matches_separate_hard_and_soft_quantization() -> None:
    torch.manual_seed(7)
    projected = torch.randn(1, 32, 2, 3, 4, requires_grad=True)
    codebook = torch.randn(4096, 32)
    temperature = 0.1

    output = quantize_with_relaxed_probabilities(
        projected, codebook, temperature=temperature
    )

    normalized = F.normalize(projected, p=2, dim=1)
    embeddings = F.normalize(codebook, p=2, dim=1)
    flat = normalized.permute(0, 2, 3, 4, 1).reshape(-1, 32)
    reference_distances = (
        flat.square().sum(dim=1, keepdim=True)
        + embeddings.square().sum(dim=1).unsqueeze(0)
        - 2.0 * flat @ embeddings.t()
    )
    reference_indices = reference_distances.argmin(dim=1).reshape(1, 2, 3, 4)
    reference_probabilities = torch.softmax(
        -reference_distances / temperature, dim=-1
    ).reshape(1, 2, 3, 4, 4096)

    assert torch.equal(output.indices, reference_indices)
    assert torch.allclose(output.target_probabilities, reference_probabilities)
    assert torch.allclose(
        output.target_probabilities.sum(dim=-1),
        torch.ones_like(output.target_probabilities[..., 0]),
    )

    soft_objective = (
        output.target_probabilities
        * torch.linspace(0.0, 1.0, 4096).reshape(1, 1, 1, 1, 4096)
    ).sum()
    (output.quantized.square().mean() + soft_objective).backward()
    assert projected.grad is not None
    assert torch.isfinite(projected.grad).all()
