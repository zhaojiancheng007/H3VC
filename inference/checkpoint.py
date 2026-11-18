"""Construct H3VC and load inference weights."""
import torch
from entropy_model.selection import build_model, config_from_checkpoint
from tokenizer.model import Tokenizer
from tokenizer.backbone import MiniMaxH3Backbone
from tokenizer.gaussian_tile_decode import enable_gaussian_tile_decode


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    config = config_from_checkpoint(checkpoint)
    model = torch.nn.Module()
    model.tokenizer = Tokenizer(MiniMaxH3Backbone(),
                                config.codebook_size, config.codebook_dim)
    model.entropy_model = build_model(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    enable_gaussian_tile_decode(model.tokenizer)
    return model
