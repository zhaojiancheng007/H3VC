"""Load the final weights and the matching entropy architecture."""
import torch
from entropy_model.selection import build_model, config_from_checkpoint, architecture
from tokenizer.model import Tokenizer
from tokenizer.backbone import MiniMaxH3Backbone
from tokenizer.gaussian_tile_decode import enable_gaussian_tile_decode


def load_model(args):
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if checkpoint["stage"] != 3 or checkpoint["tokenizer"] != "minimax_h3":
        raise ValueError("Expected a MiniMax-H3 Stage3 checkpoint")
    stage2 = torch.load(args.entropy_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    config = config_from_checkpoint(stage2)
    del stage2
    model = torch.nn.Module()
    model.tokenizer = Tokenizer(MiniMaxH3Backbone(args.minimax_h3_model_dir),
                                config.codebook_size, config.codebook_dim)
    model.entropy_model = build_model(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    enable_gaussian_tile_decode(model.tokenizer)
    return model, int(checkpoint["step"]), architecture(config), device
