import pytest
import torch
from torch import nn

from metrics.lpips import sequence_lpips


class MeanAbsoluteDistance(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(
        self,
        reference: torch.Tensor,
        reconstruction: torch.Tensor,
        *,
        normalize: bool,
    ) -> torch.Tensor:
        assert normalize is False
        self.batch_sizes.append(reference.shape[0])
        return (reference - reconstruction).abs().mean(dim=(1, 2, 3), keepdim=True)


def test_sequence_lpips_averages_one_score_per_frame() -> None:
    reference = torch.zeros(3, 3, 2, 2)
    reconstruction = torch.stack(
        [
            torch.zeros(3, 2, 2),
            torch.full((3, 2, 2), 0.25),
            torch.full((3, 2, 2), 0.5),
        ]
    )
    metric = MeanAbsoluteDistance()

    value = sequence_lpips(
        reference,
        reconstruction,
        metric,
        device="cpu",
        batch_size=2,
    )

    assert value == pytest.approx(0.5)
    assert metric.batch_sizes == [2, 1]


def test_sequence_lpips_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        sequence_lpips(
            torch.zeros(1, 3, 2, 2),
            torch.zeros(2, 3, 2, 2),
            MeanAbsoluteDistance(),
            device="cpu",
        )


def test_sequence_lpips_rejects_values_outside_unit_range() -> None:
    with pytest.raises(ValueError, match="must be in"):
        sequence_lpips(
            torch.zeros(1, 3, 2, 2),
            torch.full((1, 3, 2, 2), 1.01),
            MeanAbsoluteDistance(),
            device="cpu",
        )
