from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import struct


MAGIC = b"ARANS\x00\x02\x00"
HEADER_LENGTH = struct.Struct("<I")
STREAM_LENGTH = struct.Struct("<Q")
STREAM_COUNT = 6


@dataclass(frozen=True)
class AREntropyStreamHeader:
    tokenizer_model_id: str
    entropy_model_sha256: str
    input_shape: tuple[int, int, int]
    padded_shape: tuple[int, int, int]
    latent_shape: tuple[int, int, int]
    fsq_levels: tuple[int, ...]
    window_size: int
    cdf_precision: int = 16
    stream_count: int = STREAM_COUNT
    scan_order: str = "time-group-window-site-v1"


def _validate_header(header: AREntropyStreamHeader) -> None:
    if header.stream_count != STREAM_COUNT:
        raise ValueError(f"Expected {STREAM_COUNT} streams, got {header.stream_count}")
    if header.fsq_levels != (8, 8, 8, 5, 5, 5):
        raise ValueError(f"Unsupported FSQ levels: {header.fsq_levels}")
    if header.cdf_precision != 16:
        raise ValueError(f"Unsupported CDF precision: {header.cdf_precision}")
    if header.scan_order != "time-group-window-site-v1":
        raise ValueError(f"Unsupported scan order: {header.scan_order}")
    if len(header.entropy_model_sha256) != 64:
        raise ValueError("entropy_model_sha256 must contain 64 hexadecimal characters")
    try:
        int(header.entropy_model_sha256, 16)
    except ValueError as exc:
        raise ValueError("entropy_model_sha256 is not hexadecimal") from exc
    if any(value <= 0 for shape in (header.input_shape, header.padded_shape, header.latent_shape) for value in shape):
        raise ValueError(f"Stream shapes must be positive: {header}")


def write_ar_entropy_stream(
    path: str | Path,
    header: AREntropyStreamHeader,
    streams: tuple[bytes, ...],
) -> dict[str, int]:
    _validate_header(header)
    if len(streams) != STREAM_COUNT or any(not isinstance(stream, bytes) for stream in streams):
        raise TypeError(f"Expected {STREAM_COUNT} byte streams")
    path = Path(path)
    header_bytes = json.dumps(asdict(header), sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(HEADER_LENGTH.pack(len(header_bytes)))
        handle.write(header_bytes)
        for stream in streams:
            handle.write(STREAM_LENGTH.pack(len(stream)))
            handle.write(stream)
    return {
        "payload_bytes": sum(len(stream) for stream in streams),
        "container_bytes": path.stat().st_size,
    }


def read_ar_entropy_stream(
    path: str | Path,
) -> tuple[tuple[bytes, ...], AREntropyStreamHeader]:
    path = Path(path)
    with path.open("rb") as handle:
        if handle.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"Invalid AR entropy stream magic: {path}")
        packed_header_length = handle.read(HEADER_LENGTH.size)
        if len(packed_header_length) != HEADER_LENGTH.size:
            raise ValueError(f"Truncated entropy stream header length: {path}")
        header_length = HEADER_LENGTH.unpack(packed_header_length)[0]
        header_raw = handle.read(header_length)
        if len(header_raw) != header_length:
            raise ValueError(f"Truncated entropy stream header: {path}")
        data = json.loads(header_raw)
        header = AREntropyStreamHeader(
            tokenizer_model_id=data["tokenizer_model_id"],
            entropy_model_sha256=data["entropy_model_sha256"],
            input_shape=tuple(data["input_shape"]),
            padded_shape=tuple(data["padded_shape"]),
            latent_shape=tuple(data["latent_shape"]),
            fsq_levels=tuple(data["fsq_levels"]),
            window_size=int(data["window_size"]),
            cdf_precision=int(data["cdf_precision"]),
            stream_count=int(data["stream_count"]),
            scan_order=data["scan_order"],
        )
        _validate_header(header)
        streams = []
        for dimension in range(STREAM_COUNT):
            packed_stream_length = handle.read(STREAM_LENGTH.size)
            if len(packed_stream_length) != STREAM_LENGTH.size:
                raise ValueError(f"Missing stream {dimension} length: {path}")
            stream_length = STREAM_LENGTH.unpack(packed_stream_length)[0]
            stream = handle.read(stream_length)
            if len(stream) != stream_length:
                raise ValueError(f"Truncated stream {dimension}: {path}")
            streams.append(stream)
        if handle.read(1):
            raise ValueError(f"Trailing bytes after {STREAM_COUNT} streams: {path}")
    return tuple(streams), header
