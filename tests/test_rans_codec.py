import math

import torch

from codec.rans_codec import (
    ParallelFSQRansCodec,
    logits_to_cdfs,
)
from entropy_model import CosmosFSQEntropyModel, EntropyModelConfig


def _tiny_model() -> CosmosFSQEntropyModel:
    torch.manual_seed(19)
    return CosmosFSQEntropyModel(
        EntropyModelConfig(
            window_size=8,
            d_model=32,
            num_layers=2,
            num_heads=4,
            ffn_dim=64,
            max_latent_frames=2,
        )
    ).eval()


def test_logits_cdfs_reserve_compressai_escape_symbol() -> None:
    cdfs = logits_to_cdfs(torch.zeros(2, 8), precision=16)

    assert len(cdfs) == 2
    assert all(len(cdf) == 10 for cdf in cdfs)
    assert all(cdf[-1] == 1 << 16 for cdf in cdfs)
    assert all(cdf[-1] - cdf[-2] == 1 for cdf in cdfs)


def test_six_stream_rans_round_trip() -> None:
    model = _tiny_model()
    indices = torch.randint(0, 64000, (1, 2, 32, 32), dtype=torch.int64)
    codec = ParallelFSQRansCodec(model)

    encoded = codec.compress(indices)
    restored = codec.decompress(encoded.streams, latent_shape=(2, 32, 32))

    assert len(encoded.streams) == 6
    assert all(isinstance(stream, bytes) and stream for stream in encoded.streams)
    assert encoded.stage_count == 8
    assert encoded.symbol_count == indices.numel() * 6
    assert encoded.estimated_bits > 0
    expected_bits = float(model(indices).estimated_bits.sum().item())
    assert math.isclose(
        encoded.estimated_bits, expected_bits, rel_tol=1e-5, abs_tol=1e-3
    )
    payload_bits = sum(len(stream) * 8 for stream in encoded.streams)
    assert abs(payload_bits - encoded.estimated_bits) < 512
    assert torch.equal(restored, indices)
