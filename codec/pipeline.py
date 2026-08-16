from __future__ import annotations

from pathlib import Path
import time

import torch

from codec.index_stream import IndexStreamHeader, read_index_stream, write_index_stream
from metrics.psnr import sequence_psnr
from tokenizer.cosmos_dv.preprocessing import pad_for_cosmos, unpad_reconstruction


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, function):
    _sync(device)
    start = time.perf_counter()
    result = function()
    _sync(device)
    return result, (time.perf_counter() - start) * 1000.0


def run_segment(
    tokenizer,
    video_uint8: torch.Tensor,
    stream_path: str | Path,
) -> tuple[torch.Tensor, dict]:
    """Round-trip one [T,C,H,W] uint8 segment through a real index file."""
    if video_uint8.ndim != 4 or video_uint8.shape[1] != 3 or video_uint8.dtype != torch.uint8:
        raise ValueError(
            f"Expected uint8 [T,3,H,W], got {video_uint8.dtype} {tuple(video_uint8.shape)}"
        )
    reference = video_uint8.float().div(255.0)
    model_input = reference.mul(2.0).sub(1.0).permute(1, 0, 2, 3).unsqueeze(0)
    padded, pad_info = pad_for_cosmos(model_input)

    (indices, _codes), encode_ms = _timed(tokenizer.device, lambda: tokenizer.encode(padded))
    indices_cpu = indices.to("cpu", dtype=torch.int64)
    header = IndexStreamHeader(
        model_id=tokenizer.model_id,
        input_shape=(pad_info.frames, pad_info.height, pad_info.width),
        padded_shape=(pad_info.padded_frames, pad_info.padded_height, pad_info.padded_width),
        latent_shape=tuple(map(int, indices_cpu.shape[1:])),
        index_count=indices_cpu.numel(),
    )
    stream_stats, write_ms = _timed(torch.device("cpu"), lambda: write_index_stream(stream_path, indices_cpu, header))
    (decoded_indices, decoded_header), read_ms = _timed(
        torch.device("cpu"), lambda: read_index_stream(stream_path)
    )
    if decoded_header != header:
        raise RuntimeError("Index stream header changed during round-trip")
    if not torch.equal(decoded_indices, indices_cpu):
        raise RuntimeError("Index tensor changed during stream round-trip")

    reconstruction_padded, decode_ms = _timed(
        tokenizer.device, lambda: tokenizer.decode(decoded_indices)
    )
    reconstruction = unpad_reconstruction(reconstruction_padded, pad_info)
    reconstruction = reconstruction.squeeze(0).permute(1, 0, 2, 3)
    reconstruction = reconstruction.float().add(1.0).div(2.0).clamp(0.0, 1.0).cpu()
    pixels = pad_info.frames * pad_info.height * pad_info.width
    metrics = {
        "psnr": sequence_psnr(reference, reconstruction),
        "bpp": stream_stats["container_bytes"] * 8.0 / pixels,
        "payload_bpp": stream_stats["payload_bytes"] * 8.0 / pixels,
        "latency_ms": {
            "encode": encode_ms,
            "stream_write": write_ms,
            "stream_read": read_ms,
            "decode": decode_ms,
            "total": encode_ms + write_ms + read_ms + decode_ms,
        },
        "input_shape": list(header.input_shape),
        "padded_shape": list(header.padded_shape),
        "latent_shape": list(header.latent_shape),
        "index_min": int(indices_cpu.min()),
        "index_max": int(indices_cpu.max()),
        **stream_stats,
    }
    return reconstruction, metrics
