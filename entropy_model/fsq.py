from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch


_INTEGER_DTYPES = {
    torch.uint8,
    torch.uint16,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


@dataclass(frozen=True)
class FSQSpec:
    levels: tuple[int, ...] = (8, 8, 8, 5, 5, 5)

    def __post_init__(self) -> None:
        if not self.levels or any(level < 2 for level in self.levels):
            raise ValueError(f"Every FSQ level must be at least 2, got {self.levels}")

    @property
    def dimensions(self) -> int:
        return len(self.levels)

    @property
    def codebook_size(self) -> int:
        return prod(self.levels)

    @property
    def basis(self) -> tuple[int, ...]:
        values = [1]
        for level in self.levels[:-1]:
            values.append(values[-1] * level)
        return tuple(values)

    def _require_integer(self, tensor: torch.Tensor, name: str) -> None:
        if tensor.dtype not in _INTEGER_DTYPES:
            raise TypeError(f"{name} must use an integer dtype, got {tensor.dtype}")

    def validate_indices(self, indices: torch.Tensor) -> None:
        self._require_integer(indices, "indices")
        if indices.numel() == 0:
            raise ValueError("indices must not be empty")
        minimum = int(indices.min().item())
        maximum = int(indices.max().item())
        if minimum < 0 or maximum >= self.codebook_size:
            raise ValueError(
                f"FSQ indices must be in [0,{self.codebook_size - 1}], got min={minimum}, max={maximum}"
            )

    def indices_to_digits(self, indices: torch.Tensor) -> torch.Tensor:
        self.validate_indices(indices)
        basis = torch.tensor(self.basis, dtype=torch.int64, device=indices.device)
        levels = torch.tensor(self.levels, dtype=torch.int64, device=indices.device)
        return (indices.to(torch.int64).unsqueeze(-1) // basis) % levels

    def validate_digits(self, digits: torch.Tensor) -> None:
        self._require_integer(digits, "digits")
        if digits.ndim == 0 or digits.shape[-1] != self.dimensions:
            raise ValueError(
                f"digits must end in {self.dimensions} FSQ dimensions, got {tuple(digits.shape)}"
            )
        for dimension, level in enumerate(self.levels):
            values = digits[..., dimension]
            minimum = int(values.min().item())
            maximum = int(values.max().item())
            if minimum < 0 or maximum >= level:
                raise ValueError(
                    f"digits[...,{dimension}] must be in [0,{level - 1}], got min={minimum}, max={maximum}"
                )

    def digits_to_indices(self, digits: torch.Tensor) -> torch.Tensor:
        self.validate_digits(digits)
        basis = torch.tensor(self.basis, dtype=torch.int64, device=digits.device)
        return (digits.to(torch.int64) * basis).sum(dim=-1)

    def digits_to_codes(
        self, digits: torch.Tensor, *, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        self.validate_digits(digits)
        half_width = torch.tensor(
            [level // 2 for level in self.levels], dtype=dtype, device=digits.device
        )
        return (digits.to(dtype) - half_width) / half_width

    def indices_to_codes(
        self, indices: torch.Tensor, *, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        return self.digits_to_codes(self.indices_to_digits(indices), dtype=dtype)
