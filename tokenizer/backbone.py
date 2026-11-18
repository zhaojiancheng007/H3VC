"""Original MiniMax-H3 encode/decode path for 107-frame inference."""
from pathlib import Path
import sys
import torch
from torch import nn


class MiniMaxH3Backbone(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        model_dir = Path(model_dir).resolve()
        sys.path.insert(0, str(model_dir.parent))
        from video_vae.minimax_h3_video_vae import MiniMaxH3VideoVAE
        self.model = MiniMaxH3VideoVAE.from_pretrained(str(model_dir)).model

    def encode(self, video):
        pixels = video.float().add(1.0).div(2.0)
        normalized = self.model.processor.transform_tensor(pixels)
        moments = self.model.encode_temporal(normalized)
        posterior_mean, _ = moments.chunk(2, dim=1)
        return posterior_mean

    def decode(self, latent):
        decoded = self.model.decode_base(latent.float(), frame_num=107, process_image=False)
        pixels = self.model.processor.revert_tensor(decoded)
        return pixels.mul(2.0).sub(1.0)
