from pathlib import Path

import pytest
import torch

from codec.index_stream import IndexStreamHeader, read_index_stream, write_index_stream


def test_index_stream_round_trip(tmp_path: Path) -> None:
    indices = torch.arange(18, dtype=torch.int64).reshape(1, 2, 3, 3)
    header = IndexStreamHeader("test-model", (5, 24, 24), (9, 32, 32), (2, 3, 3), 18)
    stats = write_index_stream(tmp_path / "tokens.aridx", indices, header)
    restored, restored_header = read_index_stream(tmp_path / "tokens.aridx")
    assert torch.equal(restored, indices)
    assert restored_header == header
    assert stats["payload_bytes"] == 36
    assert stats["container_bytes"] > stats["payload_bytes"]


def test_index_stream_rejects_uint16_overflow(tmp_path: Path) -> None:
    indices = torch.tensor([[[[65536]]]])
    header = IndexStreamHeader("test", (1, 8, 8), (1, 16, 16), (1, 1, 1), 1)
    with pytest.raises(ValueError, match="fit uint16"):
        write_index_stream(tmp_path / "bad.aridx", indices, header)
