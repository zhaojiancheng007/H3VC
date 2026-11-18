# H3VC adapter for MiniMax-H3: https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/FL2VA/video_vae
# Upstream VAE license: Apache-2.0; see h3/LICENSE.
"""MiniMax-H3 construction and inference; weights come from H3VC."""
from torch import nn
from .h3.klvae import AutoencoderKLLegacy
from .h3.parallel import get_parallel_state
from .h3.config import H3_CONFIG


class MiniMaxH3Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        get_parallel_state().update(
            group_size=1, group_rank=0, local_process_group=None,
            sp_size=1, sp_rank=0, sp_enabled=False, sp_process_group=None,
            tp_size=1, tp_rank=0,
        )
        self.model = AutoencoderKLLegacy(**H3_CONFIG)

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
