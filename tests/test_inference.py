import sys
from pathlib import Path

import pytest
import torch

from entropy_model.config_categorical import CategoricalEntropyModelConfig
from entropy_model.model_categorical import CategoricalEntropyModel
import inference.checkpoint
from inference.frames import load_png_sequence
from inference.protocol import prepare_uvg_input, finish_uvg_reconstruction
from inference.reconstruction import _estimated_bits, save_reconstruction
from tokenizer.gaussian_tile_decode import GAUSSIAN_VARIANCE


def test_frame_io_padding_rate_and_independent_imports(tmp_path):
    # Source-frame IO and temporal/spatial padding must round-trip exactly.
    video = torch.randint(0, 256, (96, 3, 17, 19), dtype=torch.uint8)
    save_reconstruction(video.float() / 255, tmp_path / 'frames')
    actual = load_png_sequence(tmp_path / 'frames')
    torch.testing.assert_close(actual, video)
    prepared = prepare_uvg_input(actual.float() / 255)
    assert prepared.model_input.shape == (1, 3, 107, 32, 32)
    result = finish_uvg_reconstruction(prepared.model_input, prepared)
    torch.testing.assert_close(result, video.float() / 255)
    assert GAUSSIAN_VARIANCE == 0.04

    # A uniform 4096-code model must cost 12 bits per VALID token, excluding padding.
    model = CategoricalEntropyModel(CategoricalEntropyModelConfig(
        codebook_size=4096, codebook_dim=24, window_size=8, d_model=32,
        num_layers=1, num_heads=4, ffn_dim=64, max_latent_frames=2,
    )).eval()
    torch.nn.init.zeros_(model.output_head.weight)
    features = torch.randn(1, 24, 2, 9, 11)
    indices = torch.randint(0, 4096, (1, 2, 9, 11))
    with torch.inference_mode():
        bits = _estimated_bits(model, features, indices, k=1)
    torch.testing.assert_close(bits, torch.tensor(indices.numel() * 12.0))
    assert not any(name.split('.')[0] in {'training', 'datasets', 'metrics'}
                   or name.startswith(('src.v1', 'src.legacy', 'src.deprecated'))
                   for name in sys.modules)
    root = Path(__file__).resolve().parents[1]
    assert {p.name for p in (root / 'scripts').glob('*.py')} == {'__init__.py', 'inference.py'}


def test_missing_frame_fails(tmp_path):
    save_reconstruction(torch.zeros(1, 3, 4, 4), tmp_path / 'frames')
    with pytest.raises(FileNotFoundError):
        load_png_sequence(tmp_path / 'frames')
