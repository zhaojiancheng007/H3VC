from pathlib import Path

import pytest

from codec.ar_stream import (
    AREntropyStreamHeader,
    read_ar_entropy_stream,
    write_ar_entropy_stream,
)


def _header() -> AREntropyStreamHeader:
    return AREntropyStreamHeader(
        tokenizer_model_id="Cosmos-0.1-Tokenizer-DV4x8x8",
        entropy_model_sha256="ab" * 32,
        input_shape=(21, 1080, 1920),
        padded_shape=(21, 1088, 1920),
        latent_shape=(6, 136, 240),
        fsq_levels=(8, 8, 8, 5, 5, 5),
        window_size=8,
    )


def test_ar_entropy_stream_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "segment.arans"
    streams = tuple(bytes([dimension]) * (dimension + 1) for dimension in range(6))

    stats = write_ar_entropy_stream(path, _header(), streams)
    restored_streams, restored_header = read_ar_entropy_stream(path)

    assert restored_header == _header()
    assert restored_streams == streams
    assert stats["payload_bytes"] == sum(map(len, streams))
    assert stats["container_bytes"] == path.stat().st_size
    assert stats["container_bytes"] > stats["payload_bytes"]


def test_ar_entropy_stream_rejects_trailing_bytes(tmp_path: Path) -> None:
    path = tmp_path / "segment.arans"
    write_ar_entropy_stream(path, _header(), (b"x",) * 6)
    with path.open("ab") as handle:
        handle.write(b"unexpected")

    with pytest.raises(ValueError, match="Trailing bytes"):
        read_ar_entropy_stream(path)
