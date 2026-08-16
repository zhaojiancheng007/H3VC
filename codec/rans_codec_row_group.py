from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math

from compressai._CXX import pmf_to_quantized_cdf
from compressai.ans import BufferedRansEncoder, RansDecoder
import torch
import torch.nn.functional as F

from entropy_model.layout_row_group import RowGroupTensor
from entropy_model.model_quant_feature_row_group import CosmosFSQRowGroupEntropyModel


@dataclass(frozen=True)
class RowGroupRansEncodeResult:
    streams: tuple[bytes, ...]
    estimated_bits: float
    stage_count: int
    symbol_count: int


def logits_to_cdfs(logits: torch.Tensor, precision: int) -> list[list[int]]:
    if logits.ndim != 2:
        raise ValueError(f"Expected logits [N,L], got {tuple(logits.shape)}")
    if precision != 16:
        raise ValueError(f"Only 16-bit CDF precision is supported, got {precision}")
    logits = logits.detach().float().cpu()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite entropy logits")
    probabilities = torch.softmax(logits, dim=-1)
    escape = torch.zeros((probabilities.shape[0], 1), dtype=probabilities.dtype)
    return [
        pmf_to_quantized_cdf(row.tolist(), precision)
        for row in torch.cat((probabilities, escape), dim=-1)
    ]


def _decode_dimension(
    decoder: RansDecoder,
    cdfs: list[list[int]],
    alphabet_size: int,
) -> list[int]:
    count = len(cdfs)
    return decoder.decode_stream(
        list(range(count)),
        cdfs,
        [alphabet_size + 2] * count,
        [0] * count,
    )


class RowGroupFSQRansCodec:
    def __init__(
        self,
        model: CosmosFSQRowGroupEntropyModel,
        *,
        cdf_precision: int = 16,
        decode_workers: int = 6,
    ) -> None:
        if cdf_precision != 16:
            raise ValueError("Only cdf_precision=16 is supported")
        if decode_workers != 6:
            raise ValueError("Six independent FSQ streams require decode_workers=6")
        self.model = model.eval()
        self.cdf_precision = cdf_precision
        self.decode_workers = decode_workers

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def compress(self, indices: torch.Tensor) -> RowGroupRansEncodeResult:
        if indices.ndim != 4 or indices.shape[0] != 1:
            raise ValueError(f"Expected one index grid [1,T,H,W], got {tuple(indices.shape)}")
        indices = indices.to(self.device)
        _, frames, height, width = map(int, indices.shape)
        self.model._validate_shape(frames, height, width)
        digits = self.model.fsq.indices_to_digits(indices)
        codes = self.model.fsq.digits_to_codes(
            digits, dtype=self.model.code_projection.weight.dtype
        )
        digit_blocks = self.model.layout.to_blocks(digits)
        code_blocks = self.model.layout.to_blocks(codes)
        encoders = [BufferedRansEncoder() for _ in range(6)]
        cache = None
        estimated_bits = 0.0

        for stage in range(code_blocks.values.shape[1]):
            previous = None if stage == 0 else code_blocks.values[:, stage - 1]
            logits, cache = self.model.forward_step(
                previous,
                stage=stage,
                batch_size=1,
                latent_frames=frames,
                latent_height=height,
                latent_width=width,
                cache=cache,
            )
            for dimension, alphabet_size in enumerate(self.model.config.levels):
                flat_logits = logits[dimension].reshape(-1, alphabet_size)
                symbols_tensor = digit_blocks.values[:, stage, :, dimension].reshape(-1)
                cdfs = logits_to_cdfs(flat_logits, self.cdf_precision)
                symbols = symbols_tensor.cpu().tolist()
                count = len(symbols)
                encoders[dimension].encode_with_indexes(
                    symbols,
                    list(range(count)),
                    cdfs,
                    [alphabet_size + 2] * count,
                    [0] * count,
                )
                estimated_bits += float(
                    F.cross_entropy(flat_logits.float(), symbols_tensor, reduction="sum").item()
                    / math.log(2.0)
                )

        return RowGroupRansEncodeResult(
            streams=tuple(encoder.flush() for encoder in encoders),
            estimated_bits=estimated_bits,
            stage_count=int(code_blocks.values.shape[1]),
            symbol_count=int(indices.numel() * 6),
        )

    @torch.no_grad()
    def decompress(
        self,
        streams: tuple[bytes, ...],
        *,
        latent_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        if len(streams) != 6:
            raise ValueError(f"Expected 6 streams, got {len(streams)}")
        frames, height, width = map(int, latent_shape)
        self.model._validate_shape(frames, height, width)
        groups = height // self.model.config.rows_per_group
        stages = frames * groups
        sites = self.model.config.rows_per_group * width
        decoded = torch.empty(
            1, stages, sites, 6, dtype=torch.int64, device=self.device
        )
        decoders = [RansDecoder() for _ in range(6)]
        for decoder, stream in zip(decoders, streams, strict=True):
            decoder.set_stream(stream)

        cache = None
        previous_codes = None
        with ThreadPoolExecutor(max_workers=self.decode_workers) as executor:
            for stage in range(stages):
                logits, cache = self.model.forward_step(
                    previous_codes,
                    stage=stage,
                    batch_size=1,
                    latent_frames=frames,
                    latent_height=height,
                    latent_width=width,
                    cache=cache,
                )
                cdfs = [
                    logits_to_cdfs(
                        logits[dimension].reshape(-1, alphabet_size),
                        self.cdf_precision,
                    )
                    for dimension, alphabet_size in enumerate(self.model.config.levels)
                ]
                futures = [
                    executor.submit(
                        _decode_dimension,
                        decoders[dimension],
                        cdfs[dimension],
                        alphabet_size,
                    )
                    for dimension, alphabet_size in enumerate(self.model.config.levels)
                ]
                for dimension, future in enumerate(futures):
                    values = future.result()
                    alphabet_size = self.model.config.levels[dimension]
                    if len(values) != sites:
                        raise RuntimeError(
                            f"rANS stage {stage}, dimension {dimension} returned "
                            f"{len(values)} symbols, expected {sites}"
                        )
                    if min(values) < 0 or max(values) >= alphabet_size:
                        raise RuntimeError(
                            f"rANS stage {stage}, dimension {dimension} produced "
                            f"symbols outside [0,{alphabet_size - 1}]"
                        )
                    decoded[:, stage, :, dimension] = torch.tensor(
                        values, dtype=torch.int64, device=self.device
                    ).reshape(1, sites)
                previous_codes = self.model.fsq.digits_to_codes(
                    decoded[:, stage], dtype=self.model.code_projection.weight.dtype
                )

        digits = self.model.layout.from_blocks(
            RowGroupTensor(
                values=decoded,
                batch_size=1,
                latent_frames=frames,
                height=height,
                width=width,
                rows_per_group=self.model.config.rows_per_group,
            )
        )
        return self.model.fsq.digits_to_indices(digits)
