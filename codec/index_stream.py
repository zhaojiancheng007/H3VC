from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import struct

import numpy as np
import torch

MAGIC = b"ARIDX\x00\x01\x00"
HEADER_LENGTH = struct.Struct("<I")


@dataclass(frozen=True)
class IndexStreamHeader:
    model_id: str
    input_shape: tuple[int, int, int]
    padded_shape: tuple[int, int, int]
    latent_shape: tuple[int, int, int]
    index_count: int
    dtype: str = "uint16-le"


def write_index_stream(
    path: str | Path, indices: torch.Tensor, header: IndexStreamHeader
) -> dict[str, int]:
    path = Path(path)
    if indices.ndim != 4 or indices.shape[0] != 1:
        raise ValueError(f"Expected indices [1,T,H,W], got {tuple(indices.shape)}")
    array64 = indices.detach().to("cpu", dtype=torch.int64).contiguous().numpy()
    if array64.size != header.index_count:
        raise ValueError(f"Header count {header.index_count} != tensor count {array64.size}")
    if tuple(array64.shape[1:]) != tuple(header.latent_shape):
        raise ValueError(
            f"Header latent shape {header.latent_shape} != tensor {array64.shape[1:]}"
        )
    minimum, maximum = int(array64.min()), int(array64.max())
    if minimum < 0 or maximum > np.iinfo(np.uint16).max:
        raise ValueError(f"Indices do not fit uint16: min={minimum}, max={maximum}")

    header_bytes = json.dumps(asdict(header), sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = array64.astype("<u2", copy=False).tobytes(order="C")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(HEADER_LENGTH.pack(len(header_bytes)))
        handle.write(header_bytes)
        handle.write(payload)
    return {"payload_bytes": len(payload), "container_bytes": path.stat().st_size}


def read_index_stream(path: str | Path) -> tuple[torch.Tensor, IndexStreamHeader]:
    path = Path(path)
    with path.open("rb") as handle:
        if handle.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"Invalid ARCodec index stream magic: {path}")
        packed_length = handle.read(HEADER_LENGTH.size)
        if len(packed_length) != HEADER_LENGTH.size:
            raise ValueError(f"Truncated index stream header length: {path}")
        header_length = HEADER_LENGTH.unpack(packed_length)[0]
        header_raw = handle.read(header_length)
        if len(header_raw) != header_length:
            raise ValueError(f"Truncated index stream header: {path}")
        data = json.loads(header_raw)
        header = IndexStreamHeader(
            model_id=data["model_id"],
            input_shape=tuple(data["input_shape"]),
            padded_shape=tuple(data["padded_shape"]),
            latent_shape=tuple(data["latent_shape"]),
            index_count=int(data["index_count"]),
            dtype=data["dtype"],
        )
        if header.dtype != "uint16-le":
            raise ValueError(f"Unsupported index dtype: {header.dtype}")
        payload = handle.read()
    expected_bytes = header.index_count * 2
    if len(payload) != expected_bytes:
        raise ValueError(
            f"Payload length {len(payload)} != expected {expected_bytes}: {path}"
        )
    array = np.frombuffer(payload, dtype="<u2").copy().reshape((1, *header.latent_shape))
    return torch.from_numpy(array).to(torch.int64), header
