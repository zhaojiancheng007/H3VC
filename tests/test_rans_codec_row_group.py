import math

import torch

from codec.rans_codec_row_group import RowGroupFSQRansCodec
from entropy_model.config_row_group import RowGroupEntropyModelConfig
from entropy_model.model_quant_feature_row_group import CosmosFSQRowGroupEntropyModel


def test_row_group_six_stream_rans_round_trip() -> None:
    torch.manual_seed(53)
    model = CosmosFSQRowGroupEntropyModel(
        RowGroupEntropyModelConfig(
            rows_per_group=4,
            d_model=32,
            num_layers=2,
            num_heads=4,
            ffn_dim=64,
            max_latent_frames=2,
            max_latent_height=8,
            max_latent_width=6,
        )
    ).eval()
    indices = torch.randint(0, 64000, (1, 2, 8, 6), dtype=torch.int64)
    features = model.fsq.indices_to_codes(indices).permute(0, 4, 1, 2, 3)
    expected_bits = float(model(features, indices).estimated_bits.item())
    codec = RowGroupFSQRansCodec(model)

    encoded = codec.compress(indices)
    restored = codec.decompress(encoded.streams, latent_shape=(2, 8, 6))

    assert encoded.stage_count == 4
    assert encoded.symbol_count == indices.numel() * 6
    assert math.isclose(encoded.estimated_bits, expected_bits, rel_tol=1e-5, abs_tol=1e-3)
    assert torch.equal(restored, indices)
