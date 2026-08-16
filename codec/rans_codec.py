from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math

from compressai._CXX import pmf_to_quantized_cdf
from compressai.ans import BufferedRansEncoder, RansDecoder
import torch
import torch.nn.functional as F

from entropy_model.layout import BlockTensor
from entropy_model.model import CosmosFSQEntropyModel


@dataclass(frozen=True)
class RansEncodeResult:
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
    # CompressAI reserves the final CDF symbol for out-of-range bypass coding.
    escape = torch.zeros((probabilities.shape[0], 1), dtype=probabilities.dtype)
    coding_probabilities = torch.cat((probabilities, escape), dim=-1)
    return [
        pmf_to_quantized_cdf(row.tolist(), precision)
        for row in coding_probabilities
    ]


def _decode_dimension(
    decoder: RansDecoder,
    cdfs: list[list[int]],
    alphabet_size: int,
) -> list[int]:
    count = len(cdfs)
    indexes = list(range(count))
    lengths = [alphabet_size + 2] * count
    offsets = [0] * count
    return decoder.decode_stream(indexes, cdfs, lengths, offsets)


class ParallelFSQRansCodec:
    def __init__(
        self,
        model: CosmosFSQEntropyModel,
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
    def compress(self, indices: torch.Tensor) -> RansEncodeResult:
        if indices.ndim != 4 or indices.shape[0] != 1:
            raise ValueError(f"Expected one index grid [1,T,H,W], got {tuple(indices.shape)}")
        indices = indices.to(self.device)
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
                batch_windows=code_blocks.values.shape[0],
                cache=cache,
            )
            for dimension, alphabet_size in enumerate(self.model.config.levels):
                flat_logits = logits[dimension].reshape(-1, alphabet_size)
                symbols_tensor = digit_blocks.values[:, stage, :, dimension].reshape(-1)
                cdfs = logits_to_cdfs(flat_logits, self.cdf_precision)
                symbols = symbols_tensor.to("cpu", dtype=torch.int64).tolist()
                count = len(symbols)
                encoders[dimension].encode_with_indexes(
                    symbols,
                    list(range(count)),
                    cdfs,
                    [alphabet_size + 2] * count,
                    [0] * count,
                )
                estimated_bits += float(
                    F.cross_entropy(
                        flat_logits.float(), symbols_tensor, reduction="sum"
                    ).item()
                    / math.log(2.0)
                )

        streams = tuple(encoder.flush() for encoder in encoders)
        return RansEncodeResult(
            streams=streams,
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
        if frames <= 0 or frames > self.model.config.max_latent_frames:
            raise ValueError(f"Unsupported latent frame count: {frames}")
        if height % self.model.config.window_size or width % self.model.config.window_size:
            raise ValueError(
                f"Latent size {(height, width)} is not divisible by window_size={self.model.config.window_size}"
            )

        windows_y = height // self.model.config.window_size
        windows_x = width // self.model.config.window_size
        batch_windows = windows_y * windows_x
        stages = frames * self.model.layout.num_groups
        sites = self.model.layout.sites_per_block
        decoded_blocks = torch.empty(
            batch_windows, stages, sites, 6, dtype=torch.int64, device=self.device
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
                    batch_windows=batch_windows,
                    cache=cache,
                )
                cdfs_by_dimension = [
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
                        cdfs_by_dimension[dimension],
                        alphabet_size,
                    )
                    for dimension, alphabet_size in enumerate(self.model.config.levels)
                ]
                for dimension, future in enumerate(futures):
                    values = future.result()
                    alphabet_size = self.model.config.levels[dimension]
                    if len(values) != batch_windows * sites:
                        raise RuntimeError(
                            f"rANS stage {stage}, dimension {dimension} returned "
                            f"{len(values)} symbols, expected {batch_windows * sites}"
                        )
                    minimum = min(values)
                    maximum = max(values)
                    if minimum < 0 or maximum >= alphabet_size:
                        raise RuntimeError(
                            f"rANS symbol out of range at stage {stage}, dimension {dimension}: "
                            f"expected [0,{alphabet_size - 1}], got min={minimum}, max={maximum}"
                        )
                    decoded_blocks[:, stage, :, dimension] = torch.tensor(
                        values, dtype=torch.int64, device=self.device
                    ).reshape(batch_windows, sites)
                previous_codes = self.model.fsq.digits_to_codes(
                    decoded_blocks[:, stage],
                    dtype=self.model.code_projection.weight.dtype,
                )

        block_tensor = BlockTensor(
            decoded_blocks,
            batch_size=1,
            latent_frames=frames,
            height=height,
            width=width,
            windows_y=windows_y,
            windows_x=windows_x,
        )
        digits = self.model.layout.from_blocks(block_tensor)
        return self.model.fsq.digits_to_indices(digits)
