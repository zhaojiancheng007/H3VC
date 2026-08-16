from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .stage1_vq import (
    CODEBOOK_DIM,
    CODEBOOK_SIZE,
    EMAVectorQuantizer,
    Stage1VQTokenizer,
)


QUANTIZER_GEOMETRY = "unit_sphere_l2"


class NormalizedEMAVectorQuantizer(EMAVectorQuantizer):
    """EMA spherical k-means with unit-norm latents and code vectors."""

    @torch.no_grad()
    def _initialize(self, inputs: torch.Tensor) -> None:
        inputs = F.normalize(inputs, p=2, dim=-1)
        if dist.is_initialized():
            gathered = [torch.empty_like(inputs) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, inputs)
            pool = torch.cat(gathered, dim=0)
        else:
            pool = inputs
        if pool.shape[0] < self.codebook_size:
            raise ValueError(
                f"EMA-VQ initialization needs at least {self.codebook_size} latent "
                f"vectors, got {pool.shape[0]}"
            )
        if not dist.is_initialized() or dist.get_rank() == 0:
            centers = pool[
                torch.randperm(pool.shape[0], device=pool.device)[: self.codebook_size]
            ]
            for _ in range(10):
                distances = torch.cdist(
                    pool,
                    centers,
                    p=2,
                    compute_mode="donot_use_mm_for_euclid_dist",
                )
                nearest_distances, assignments = distances.min(dim=1)
                counts = torch.bincount(
                    assignments, minlength=self.codebook_size
                ).float()
                sums = torch.zeros_like(centers)
                sums.scatter_add_(
                    0,
                    assignments[:, None].expand(-1, self.codebook_dim),
                    pool,
                )
                occupied = counts > 0
                centers[occupied] = F.normalize(
                    sums[occupied] / counts[occupied, None], p=2, dim=-1
                )
                empty_count = int((~occupied).sum())
                if empty_count:
                    farthest = nearest_distances.topk(empty_count).indices
                    centers[~occupied] = pool[farthest]
        else:
            centers = torch.empty_like(self.embed)
        if dist.is_initialized():
            dist.broadcast(centers, src=0)
        self.embed.copy_(centers)
        self.cluster_size.fill_(1.0)
        self.embed_sum.copy_(centers)
        self.initted.fill_(True)

    @torch.no_grad()
    def _nearest_indices(self, inputs: torch.Tensor) -> torch.Tensor:
        return super()._nearest_indices(F.normalize(inputs, p=2, dim=-1))

    @torch.no_grad()
    def _update(self, inputs: torch.Tensor, indices: torch.Tensor) -> None:
        inputs = F.normalize(inputs, p=2, dim=-1)
        counts = torch.zeros(self.codebook_size, device=inputs.device)
        sums = torch.zeros_like(self.embed)
        counts.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float32))
        sums.scatter_add_(
            0,
            indices[:, None].expand(-1, self.codebook_dim),
            inputs,
        )
        if dist.is_initialized():
            dist.all_reduce(counts)
            dist.all_reduce(sums)
        self.cluster_size.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
        self.embed_sum.mul_(self.decay).add_(sums, alpha=1.0 - self.decay)
        means = self.embed_sum / self.cluster_size.clamp_min(1e-5)[:, None]
        self.embed.copy_(F.normalize(means, p=2, dim=-1))

    def forward(self, projected: torch.Tensor):
        normalized = F.normalize(projected.float(), p=2, dim=1).to(projected.dtype)
        return super().forward(normalized)


class NormalizedStage1VQTokenizer(Stage1VQTokenizer):
    def __init__(
        self,
        backbone: torch.nn.Module,
        *,
        latent_channels: int = 16,
        codebook_size: int = CODEBOOK_SIZE,
        codebook_dim: int = CODEBOOK_DIM,
    ) -> None:
        super().__init__(
            backbone,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            latent_channels=latent_channels,
        )
        self.quantizer = NormalizedEMAVectorQuantizer(codebook_size, codebook_dim)
