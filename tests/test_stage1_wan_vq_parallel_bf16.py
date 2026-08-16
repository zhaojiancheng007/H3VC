import torch
import torch.nn.functional as F

from tokenizer.wan_vq.model_standard_parallel_bf16 import (
    ParallelBF16WanVQStandardAdapter,
)


def test_stage1_vq_stays_fp32_and_backpropagates_under_autocast() -> None:
    adapter = ParallelBF16WanVQStandardAdapter.__new__(
        ParallelBF16WanVQStandardAdapter
    )
    torch.nn.Module.__init__(adapter)
    adapter.codebook_dim = 32
    adapter.codebook = torch.nn.Embedding(4096, 32)

    projected = torch.randn(1, 32, 2, 4, 4, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        quantized, normalized, selected, indices = adapter.quantize(projected)
        codebook_loss = F.mse_loss(selected, normalized.detach())
        commitment_loss = F.mse_loss(normalized, selected.detach())
        loss = quantized.square().mean() + codebook_loss + 0.25 * commitment_loss
    loss.backward()

    assert quantized.dtype == torch.float32
    assert normalized.dtype == torch.float32
    assert selected.dtype == torch.float32
    assert indices.dtype == torch.int64
    assert torch.isfinite(projected.grad).all()
    assert torch.isfinite(adapter.codebook.weight.grad).all()
