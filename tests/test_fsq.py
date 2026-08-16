import pytest
import torch

from entropy_model.fsq import FSQSpec


def test_all_cosmos_fsq_indices_round_trip() -> None:
    spec = FSQSpec()
    indices = torch.arange(spec.codebook_size, dtype=torch.int64)
    digits = spec.indices_to_digits(indices)
    codes = spec.digits_to_codes(digits)

    assert digits.shape == (64000, 6)
    assert codes.shape == (64000, 6)
    assert torch.equal(spec.digits_to_indices(digits), indices)
    assert tuple(spec.basis) == (1, 8, 64, 512, 2560, 12800)
    assert torch.equal(digits.min(dim=0).values, torch.zeros(6, dtype=torch.int64))
    assert torch.equal(
        digits.max(dim=0).values, torch.tensor([7, 7, 7, 4, 4, 4])
    )


def test_fsq_rejects_invalid_index_and_digit() -> None:
    spec = FSQSpec()
    with pytest.raises(ValueError, match="FSQ indices"):
        spec.indices_to_digits(torch.tensor([64000]))
    with pytest.raises(ValueError, match=r"digits\[...,3\]"):
        spec.digits_to_indices(torch.tensor([[0, 0, 0, 5, 0, 0]]))


def test_uint16_indices_are_supported() -> None:
    spec = FSQSpec()
