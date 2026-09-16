import importlib
import sys
from pathlib import Path

import pytest
import torch

from entropy_model.selection import build_model, config_from_checkpoint, LOCAL, GLOBAL_LOCAL
from entropy_model.config_categorical import CategoricalEntropyModelConfig
from entropy_model.config_local_global_memory import LocalGlobalMemoryEntropyConfig
from scripts.stage3 import EntropyFloorJointVQModel, entropy_floor_penalty_bpp
from tokenizer.stage1_vq import Stage1VQTokenizer
from training.config_stage2_local_global_memory import load_stage2_config
from training.config_stage3_cosine import load_stage3_config

ROOT = Path(__file__).resolve().parents[1]


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Conv3d(3, 24, 1)
        self.decoder = torch.nn.Conv3d(24, 3, 1)

    def encode(self, video):
        return self.encoder(video)

    def decode(self, latent):
        return self.decoder(latent)


@pytest.mark.parametrize('config_type', [CategoricalEntropyModelConfig, LocalGlobalMemoryEntropyConfig])
def test_joint_forward_backward(config_type):
    torch.manual_seed(7)
    tokenizer = Stage1VQTokenizer(TinyBackbone(), latent_channels=24, codebook_dim=24)
    tokenizer.quantizer.embed.normal_(std=0.1)
    tokenizer.quantizer.initted.fill_(True)
    config = config_type(codebook_dim=24, window_size=8, d_model=16, num_heads=2,
                         num_layers=1, ffn_dim=32, max_latent_frames=4)
    entropy = build_model(config)
    torch.nn.init.normal_(entropy.output_head.weight, std=0.01)
    model = EntropyFloorJointVQModel(tokenizer, entropy)
    video = torch.randn(1, 3, 2, 8, 8)
    output = model(video, temperature=0.1)
    loss = (output.tokenizer.reconstruction-video).square().mean() + output.entropy.soft_estimated_bits.mean()/128
    loss.backward()
    assert torch.isfinite(loss)
    assert tokenizer.backbone.encoder.weight.grad.norm() > 0
    assert entropy.output_head.weight.grad.norm() > 0
    assert tokenizer.quantizer.embed.grad is None


def test_floor_and_config():
    entropy = torch.tensor(5.0, requires_grad=True)
    penalty = entropy_floor_penalty_bpp(entropy, floor_bits=6.0, multiplier=1.0, latent_density=0.01)
    penalty.backward()
    assert penalty.item() == pytest.approx(0.01)
    assert entropy.grad.item() == pytest.approx(-0.01)
    assert load_stage2_config(ROOT/'configs/stage2_local.yaml').model.codebook_dim == 24
    assert load_stage2_config(ROOT/'configs/stage2_local_global_memory.yaml').model.memory_tokens == 512
    assert load_stage3_config(ROOT/'configs/stage3.yaml').loss.temperature == 0.1
    with pytest.raises(ValueError):
        config_from_checkpoint({'architecture': 'unsupported', 'version': 1})


def test_independent_imports():
    for entry in ('stage1', 'stage2', 'stage3', 'evaluate', 'inference'):
        importlib.import_module('scripts.' + entry)
    for name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if name.split('.')[0] in {'scripts','training','tokenizer','entropy_model','metrics','datasets'} and filename:
            assert Path(filename).resolve().is_relative_to(ROOT)
        assert not any(x in name for x in ('src.v1', 'src.legacy', 'src.deprecated'))


def test_relative_dataset_and_path_boundary(tmp_path):
    import json
    from datasets.trainable_jsons_sequence_fast import SeededFastTrainableJsonsVideoDataset
    data = tmp_path / 'videos'
    metadata = tmp_path / 'metadata'
    data.mkdir()
    metadata.mkdir()
    path = metadata / 'valid_all_chunk_0.json'
    record = {'video_path': 'clip.mp4', 'video_length': 120, 'video_height': 1080, 'video_width': 1920}
    path.write_text(json.dumps({'clips': [record]}))
    dataset = SeededFastTrainableJsonsVideoDataset(data, metadata, num_frames=107)
    assert dataset.video_paths == [data/'clip.mp4']
    record['video_path'] = '../outside.mp4'
    path.write_text(json.dumps({'clips': [record]}))
    with pytest.raises(ValueError, match='Unsafe video path'):
        SeededFastTrainableJsonsVideoDataset(data, metadata, num_frames=107)
