"""H3 tokenizer inference; parameter names match the existing checkpoints."""
import torch
from torch import nn


class Codebook(nn.Module):
    def __init__(self, size, dim):
        super().__init__()
        self.chunk_size = 4096
        self.register_buffer("embed", torch.zeros(size, dim))

    def _nearest_indices(self, inputs: torch.Tensor) -> torch.Tensor:
        indices = torch.empty(inputs.shape[0], device=inputs.device, dtype=torch.long)
        for start in range(0, inputs.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, inputs.shape[0])
            indices[start:end] = torch.cdist(
                inputs[start:end],
                self.embed,
                p=2,
                compute_mode="donot_use_mm_for_euclid_dist",
            ).argmin(dim=1)
        return indices


class Tokenizer(nn.Module):
    def __init__(self, backbone, codebook_size, codebook_dim):
        super().__init__()
        self.backbone = backbone
        self.pre_quant = nn.Conv3d(24, codebook_dim, 1)
        self.quantizer = Codebook(codebook_size, codebook_dim)
        self.post_quant = nn.Conv3d(codebook_dim, 24, 1)

    def encode(self, video):
        return self.backbone.encode(video)

    def decode(self, latent):
        return self.backbone.decode(latent)
