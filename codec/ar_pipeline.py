from __future__ import annotations

from pathlib import Path
import time

import torch

from codec.ar_stream import (
    AREntropyStreamHeader,
    read_ar_entropy_stream,
    write_ar_entropy_stream,
)
from codec.rans_codec import ParallelFSQRansCodec
from metrics.psnr import sequence_psnr
from tokenizer.cosmos_dv.preprocessing import pad_for_cosmos, unpad_reconstruction


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, function):
    _sync(device)
    started = time.perf_counter()
    result = function()
    _sync(device)
    return result, (time.perf_counter() - started) * 1000.0


def run_ar_segment(
    tokenizer,
    entropy_model,
    entropy_model_sha256: str,
    video_uint8: torch.Tensor,
    stream_path: str | Path,
) -> tuple[torch.Tensor, dict]:
    if video_uint8.ndim != 4 or video_uint8.shape[1] != 3 or video_uint8.dtype != torch.uint8:
        raise ValueError(
            f"Expected uint8 [T,3,H,W], got {video_uint8.dtype} {tuple(video_uint8.shape)}"
        )
    reference = video_uint8.float().div(255.0)
    model_input = reference.mul(2.0).sub(1.0).permute(1, 0, 2, 3).unsqueeze(0)
    padded, pad_info = pad_for_cosmos(model_input, temporal_alignment=4)

    (indices, official_codes), tokenizer_encode_ms = _timed(
        tokenizer.device, lambda: tokenizer.encode(padded)
    )
    restored_codes = entropy_model.fsq.indices_to_codes(
        indices, dtype=official_codes.dtype
    ).permute(0, 4, 1, 2, 3)
    if not torch.equal(restored_codes, official_codes):
        raise RuntimeError("Cosmos codes disagree with indices_to_codes during AR encoding")

    rans = ParallelFSQRansCodec(entropy_model)
    encoded, entropy_encode_ms = _timed(
        rans.device, lambda: rans.compress(indices)
    )
    header = AREntropyStreamHeader(
        tokenizer_model_id=tokenizer.model_id,
        entropy_model_sha256=entropy_model_sha256,
        input_shape=(pad_info.frames, pad_info.height, pad_info.width),
        padded_shape=(
            pad_info.padded_frames,
            pad_info.padded_height,
            pad_info.padded_width,
        ),
        latent_shape=tuple(map(int, indices.shape[1:])),
        fsq_levels=tuple(entropy_model.config.levels),
        window_size=entropy_model.config.window_size,
    )
    stream_stats, stream_write_ms = _timed(
        torch.device("cpu"),
        lambda: write_ar_entropy_stream(stream_path, header, encoded.streams),
    )
    (streams, decoded_header), stream_read_ms = _timed(
        torch.device("cpu"), lambda: read_ar_entropy_stream(stream_path)
    )
    if decoded_header != header:
        raise RuntimeError("AR entropy stream header changed during round-trip")

    decoded_indices, entropy_decode_ms = _timed(
        rans.device,
        lambda: rans.decompress(streams, latent_shape=decoded_header.latent_shape),
    )
    if not torch.equal(decoded_indices.to(indices.device), indices.to(torch.int64)):
        raise RuntimeError("rANS round-trip changed Cosmos indices")

    reconstruction_padded, tokenizer_decode_ms = _timed(
        tokenizer.device, lambda: tokenizer.decode(decoded_indices)
    )
    reconstruction = unpad_reconstruction(reconstruction_padded, pad_info)
    reconstruction = reconstruction.squeeze(0).permute(1, 0, 2, 3)
    reconstruction = reconstruction.float().add(1.0).div(2.0).clamp(0.0, 1.0).cpu()
    pixels = pad_info.frames * pad_info.height * pad_info.width
    latency = {
        "tokenizer_encode": tokenizer_encode_ms,
        "entropy_encode": entropy_encode_ms,
        "stream_write": stream_write_ms,
        "stream_read": stream_read_ms,
        "entropy_decode": entropy_decode_ms,
        "tokenizer_decode": tokenizer_decode_ms,
    }
    latency["total"] = sum(latency.values())
    metrics = {
        "psnr": sequence_psnr(reference, reconstruction),
        "bpp": stream_stats["container_bytes"] * 8.0 / pixels,
        "payload_bpp": stream_stats["payload_bytes"] * 8.0 / pixels,
        "estimated_bpp": encoded.estimated_bits / pixels,
        "estimated_bits": encoded.estimated_bits,
        "latency_ms": latency,
        "input_shape": list(header.input_shape),
        "padded_shape": list(header.padded_shape),
        "latent_shape": list(header.latent_shape),
        "stage_count": encoded.stage_count,
        "symbol_count": encoded.symbol_count,
        **stream_stats,
    }
    return reconstruction, metrics
