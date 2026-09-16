from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


CODEBOOK_SIZE = 4096
CODEBOOK_DIM = 16
QUANTIZER_GEOMETRY = "raw_l2"
DEAD_CODE_THRESHOLD = 1.0
DEAD_CODE_UPDATE_INTERVAL = 1
DEAD_CODE_RESET_CLUSTER_SIZE = 1.0
DEAD_CODE_SAMPLES_PER_RANK = 512


@dataclass
class QuantizedLatent:
    decoder_latent: torch.Tensor
    indices: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor


@dataclass
class Stage1TokenizerOutput:
    reconstruction: torch.Tensor
    indices: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor


class EMAVectorQuantizer(nn.Module):
    """Single 4096-entry L2 codebook updated as global online k-means."""

    def __init__(
        self,
        codebook_size: int = CODEBOOK_SIZE,
        codebook_dim: int = CODEBOOK_DIM,
        *,
        decay: float = 0.99,
        commitment_weight: float = 0.25,
        chunk_size: int = 4096,
        dead_code_threshold: float = DEAD_CODE_THRESHOLD,
        dead_code_update_interval: int = DEAD_CODE_UPDATE_INTERVAL,
        reset_cluster_size: float = DEAD_CODE_RESET_CLUSTER_SIZE,
        samples_per_rank: int = DEAD_CODE_SAMPLES_PER_RANK,
    ) -> None:
        super().__init__()
        if dead_code_threshold <= 0.0:
            raise ValueError("dead_code_threshold must be positive")
        if dead_code_update_interval <= 0:
            raise ValueError("dead_code_update_interval must be positive")
        if reset_cluster_size <= 0.0:
            raise ValueError("reset_cluster_size must be positive")
        if samples_per_rank <= 0:
            raise ValueError("samples_per_rank must be positive")
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.commitment_weight = commitment_weight
        self.chunk_size = chunk_size
        self.dead_code_threshold = dead_code_threshold
        self.dead_code_update_interval = dead_code_update_interval
        self.reset_cluster_size = reset_cluster_size
        self.samples_per_rank = samples_per_rank
        self.register_buffer("embed", torch.zeros(codebook_size, codebook_dim))
        self.register_buffer("cluster_size", torch.ones(codebook_size))
        self.register_buffer("embed_sum", torch.zeros(codebook_size, codebook_dim))
        self.register_buffer("initted", torch.tensor(False))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        update_key = prefix + "num_updates"
        if update_key not in state_dict:
            state_dict[update_key] = self.num_updates.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.no_grad()
    def _initialize(self, inputs: torch.Tensor) -> None:
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
                centers[occupied] = sums[occupied] / counts[occupied, None]
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

    @torch.no_grad()
    def _update(self, inputs: torch.Tensor, indices: torch.Tensor) -> None:
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
        self.embed.copy_(self.embed_sum / self.cluster_size.clamp_min(1e-5)[:, None])

    @torch.no_grad()
    def _sample_pool(self, inputs: torch.Tensor) -> torch.Tensor:
        indices = torch.randint(
            0,
            inputs.shape[0],
            (self.samples_per_rank,),
            device=inputs.device,
        )
        local_samples = inputs.index_select(0, indices)
        if not dist.is_initialized():
            return local_samples
        gathered = [
            torch.empty_like(local_samples) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered, local_samples)
        return torch.cat(gathered, dim=0)

    @torch.no_grad()
    def _revive_dead_codes_(self, inputs: torch.Tensor) -> None:
        if int(self.num_updates.item()) % self.dead_code_update_interval != 0:
            return
        dead_ids = torch.nonzero(
            self.cluster_size < self.dead_code_threshold,
            as_tuple=False,
        ).squeeze(1)
        if dead_ids.numel() == 0:
            return

        pool = self._sample_pool(inputs)
        dead_count = int(dead_ids.numel())
        if not dist.is_initialized() or dist.get_rank() == 0:
            selected = torch.randint(
                0,
                pool.shape[0],
                (dead_count,),
                device=pool.device,
            )
            replacements = pool.index_select(0, selected)
        else:
            replacements = torch.empty(
                dead_count,
                self.codebook_dim,
                device=inputs.device,
                dtype=torch.float32,
            )
        if dist.is_initialized():
            dist.broadcast(replacements, src=0)

        self.embed[dead_ids] = replacements
        self.cluster_size[dead_ids] = self.reset_cluster_size
        self.embed_sum[dead_ids] = replacements * self.reset_cluster_size

    def forward(
        self, projected: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = projected.dtype
        shape = projected.shape
        flat = (
            projected.permute(0, 2, 3, 4, 1)
            .reshape(-1, self.codebook_dim)
            .float()
            .contiguous()
        )
        if self.training and not bool(self.initted.item()):
            self._initialize(flat.detach())
        if not bool(self.initted.item()):
            raise RuntimeError("EMA-VQ codebook has not been initialized")

        indices = self._nearest_indices(flat)
        selected = self.embed.index_select(0, indices)
        commitment = F.mse_loss(flat, selected.detach())
        codebook_error = F.mse_loss(selected, flat.detach())
        quantized = flat + (selected - flat).detach()
        if self.training:
            self._update(flat.detach(), indices)
            self._revive_dead_codes_(flat.detach())
            self.num_updates.add_(1)

        quantized = quantized.to(dtype).reshape(
            shape[0], shape[2], shape[3], shape[4], self.codebook_dim
        ).permute(0, 4, 1, 2, 3).contiguous()
        indices = indices.reshape(shape[0], shape[2], shape[3], shape[4])
        return quantized, indices, self.commitment_weight * commitment, codebook_error


class Stage1VQTokenizer(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        codebook_size: int = CODEBOOK_SIZE,
        codebook_dim: int = CODEBOOK_DIM,
        latent_channels: int = 16,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pre_quant = nn.Conv3d(latent_channels, codebook_dim, kernel_size=1)
        self.quantizer = EMAVectorQuantizer(codebook_size, codebook_dim)
        self.post_quant = nn.Conv3d(codebook_dim, latent_channels, kernel_size=1)
        with torch.no_grad():
            self.pre_quant.weight.zero_()
            self.pre_quant.bias.zero_()
            self.post_quant.weight.zero_()
            self.post_quant.bias.zero_()
            shared = min(latent_channels, codebook_dim)
            identity = torch.eye(shared)
            self.pre_quant.weight[:shared, :shared, 0, 0, 0].copy_(identity)
            self.post_quant.weight[:shared, :shared, 0, 0, 0].copy_(identity)

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode(video)

    def quantize(self, latent: torch.Tensor) -> QuantizedLatent:
        projected = self.pre_quant(latent)
        quantized, indices, commitment_loss, codebook_loss = self.quantizer(projected)
        return QuantizedLatent(
            decoder_latent=self.post_quant(quantized),
            indices=indices,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
        )

    def decode(self, decoder_latent: torch.Tensor) -> torch.Tensor:
        return self.backbone.decode(decoder_latent)

    def forward(self, video: torch.Tensor) -> Stage1TokenizerOutput:
        quantized = self.quantize(self.encode(video))
        reconstruction = self.decode(quantized.decoder_latent)
        return Stage1TokenizerOutput(
            reconstruction=reconstruction,
            indices=quantized.indices,
            commitment_loss=quantized.commitment_loss,
            codebook_loss=quantized.codebook_loss,
        )
