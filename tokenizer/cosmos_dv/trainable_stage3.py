from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


DV_LEVELS = (8, 8, 8, 5, 5, 5)


@dataclass(frozen=True)
class TrainableCosmosDVOutput:
    prequant: torch.Tensor
    bounded: torch.Tensor
    indices: torch.Tensor
    quant_features: torch.Tensor
    reconstruction: torch.Tensor


def fsq_bound(
    prequant: torch.Tensor,
    levels: tuple[int, ...] = DV_LEVELS,
) -> torch.Tensor:
    """Reproduce the FP32 bound operation embedded in the released JIT graph."""
    if prequant.ndim != 5 or prequant.shape[1] != len(levels):
        raise ValueError(
            f"Expected prequant [B,{len(levels)},T,H,W], got {tuple(prequant.shape)}"
        )
    values = prequant.permute(0, 2, 3, 4, 1)
    level_tensor = torch.tensor(levels, dtype=torch.int32, device=values.device)
    half_l = (level_tensor - 1) * (1 + 1e-3) / 2
    offset = torch.where(level_tensor % 2 == 0, 0.5, 0.0)
    shift = (offset / half_l).atanh()
    bounded = (values + shift).tanh() * half_l - offset
    return bounded.permute(0, 4, 1, 2, 3)


def fsq_surrogate_probabilities(
    bounded: torch.Tensor,
    *,
    temperature: float,
    levels: tuple[int, ...] = DV_LEVELS,
) -> tuple[torch.Tensor, ...]:
    """Return one differentiable nearest-level distribution per FSQ digit."""
    if bounded.ndim != 5 or bounded.shape[1] != len(levels):
        raise ValueError(
            f"Expected bounded FSQ values [B,{len(levels)},T,H,W], "
            f"got {tuple(bounded.shape)}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    probabilities: list[torch.Tensor] = []
    values = bounded.float().permute(0, 2, 3, 4, 1)
    for dimension, level in enumerate(levels):
        centers = torch.arange(
            level, dtype=values.dtype, device=values.device
        ) - level // 2
        squared_distance = (
            values[..., dimension].unsqueeze(-1) - centers
        ).square()
        probabilities.append(torch.softmax(-squared_distance / temperature, dim=-1))
    return tuple(probabilities)


class TrainableCosmosDV(nn.Module):
    """Train the exact BF16 modules stored in the released Cosmos JIT archives."""

    model_id = "Cosmos-0.1-Tokenizer-DV4x8x8"

    def __init__(
        self,
        encoder_checkpoint: str | Path,
        decoder_checkpoint: str | Path,
    ) -> None:
        super().__init__()
        encoder_checkpoint = Path(encoder_checkpoint)
        decoder_checkpoint = Path(decoder_checkpoint)
        if not encoder_checkpoint.is_file():
            raise FileNotFoundError(encoder_checkpoint)
        if not decoder_checkpoint.is_file():
            raise FileNotFoundError(decoder_checkpoint)

        self.encoder_jit = torch.jit.load(str(encoder_checkpoint), map_location="cpu")
        self.decoder_jit = torch.jit.load(str(decoder_checkpoint), map_location="cpu")
        if set(self.encoder_jit._modules.keys()) != {"encoder", "quant_conv", "quantizer"}:
            raise RuntimeError(
                f"Unexpected encoder JIT modules: {list(self.encoder_jit._modules.keys())}"
            )
        if set(self.decoder_jit._modules.keys()) != {"inv_quant", "post_quant_conv", "decoder"}:
            raise RuntimeError(
                f"Unexpected decoder JIT modules: {list(self.decoder_jit._modules.keys())}"
            )
        if tuple(self.encoder_jit.quantizer._levels.tolist()) != DV_LEVELS:
            raise RuntimeError(
                "Unexpected Cosmos FSQ levels: "
                f"{self.encoder_jit.quantizer._levels.tolist()}"
            )
        parameter_dtypes = {parameter.dtype for parameter in self.parameters()}
        if parameter_dtypes != {torch.bfloat16}:
            raise RuntimeError(
                f"Released Cosmos JIT parameters must be BF16, got {parameter_dtypes}"
            )

    def forward(self, video: torch.Tensor) -> TrainableCosmosDVOutput:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"Expected video [B,3,T,H,W], got {tuple(video.shape)}")
        with torch.jit.optimized_execution(False):
            prequant = self.encoder_jit.encoder(video.to(torch.bfloat16))
            prequant = self.encoder_jit.quant_conv(prequant)
            indices, quant_features, _ = self.encoder_jit.quantizer(prequant)
            bounded = fsq_bound(prequant)
            reconstruction = self.decoder_jit.post_quant_conv(quant_features)
            reconstruction = self.decoder_jit.decoder(reconstruction)
        return TrainableCosmosDVOutput(
            prequant=prequant,
            bounded=bounded,
            indices=indices,
            quant_features=quant_features,
            reconstruction=reconstruction,
        )
