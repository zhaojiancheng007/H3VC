from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from tokenizer.wan_vq.parallel_wan import parallel_decode, parallel_encode


CODEBOOK_SIZE = 4096
CODEBOOK_DIM = 16
QUANTIZER_GEOMETRY = "raw_l2"


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
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.commitment_weight = commitment_weight
        self.chunk_size = chunk_size
        self.register_buffer("embed", torch.zeros(codebook_size, codebook_dim))
        self.register_buffer("cluster_size", torch.ones(codebook_size))
        self.register_buffer("embed_sum", torch.zeros(codebook_size, codebook_dim))
        self.register_buffer("initted", torch.tensor(False))

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

        quantized = quantized.to(dtype).reshape(
            shape[0], shape[2], shape[3], shape[4], self.codebook_dim
        ).permute(0, 4, 1, 2, 3).contiguous()
        indices = indices.reshape(shape[0], shape[2], shape[3], shape[4])
        return quantized, indices, self.commitment_weight * commitment, codebook_error


class WanVAEBackbone(nn.Module):
    def __init__(self, wan_repo: str | Path, checkpoint: str | Path) -> None:
        super().__init__()
        wan_repo = Path(wan_repo).resolve()
        checkpoint = Path(checkpoint).resolve()
        sys.path.insert(0, str(wan_repo))
        from wan.modules.vae import WanVAE

        wrapper = WanVAE(
            z_dim=16,
            vae_pth=str(checkpoint),
            dtype=torch.float32,
            device="cpu",
        )
        self.model = wrapper.model.train().requires_grad_(True)
        self.register_buffer("latent_mean", wrapper.mean.detach().clone())
        self.register_buffer("latent_inv_std", wrapper.std.detach().clone().reciprocal())

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        return parallel_encode(
            self.model,
            video.float(),
            [self.latent_mean, self.latent_inv_std],
            activation_checkpointing=self.training,
        )

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return parallel_decode(
            self.model,
            latent.float(),
            [self.latent_mean, self.latent_inv_std],
            activation_checkpointing=self.training,
        )


class CosmosCV488Backbone(nn.Module):
    ENCODER_MISSING = {
        "encoder.down.0.downsample.conv2.conv3d.weight",
        "encoder.down.0.downsample.conv2.conv3d.bias",
    }
    ENCODER_UNEXPECTED = {
        "encoder.patcher3d.wavelets",
        "encoder.patcher3d._arange",
        "encoder.patcher3d.patch_size_buffer",
    }
    DECODER_MISSING = {
        "decoder.up.1.upsample.conv1.conv3d.weight",
        "decoder.up.1.upsample.conv1.conv3d.bias",
    }
    DECODER_UNEXPECTED = {
        "decoder.unpatcher3d.wavelets",
        "decoder.unpatcher3d._arange",
    }

    def __init__(
        self,
        cosmos_repo: str | Path,
        encoder_checkpoint: str | Path,
        decoder_checkpoint: str | Path,
    ) -> None:
        super().__init__()
        cosmos_repo = Path(cosmos_repo).resolve()
        sys.path.insert(0, str(cosmos_repo))
        from cosmos_tokenizer.networks import TokenizerConfigs, TokenizerModels

        config = deepcopy(TokenizerConfigs.CV.value)
        config.update(
            temporal_compression=4,
            spatial_compression=8,
            channels_mult=[2, 4],
        )
        full_model = TokenizerModels.CV.value(**config)
        self.encoder = full_model.encoder_jit()
        self.decoder = full_model.decoder_jit()

        encoder_jit = torch.jit.load(str(encoder_checkpoint), map_location="cpu")
        decoder_jit = torch.jit.load(str(decoder_checkpoint), map_location="cpu")
        encoder_result = self.encoder.load_state_dict(
            encoder_jit.state_dict(), strict=False
        )
        decoder_result = self.decoder.load_state_dict(
            decoder_jit.state_dict(), strict=False
        )
        if set(encoder_result.missing_keys) != self.ENCODER_MISSING or set(
            encoder_result.unexpected_keys
        ) != self.ENCODER_UNEXPECTED:
            raise RuntimeError(f"Unexpected Cosmos encoder state mismatch: {encoder_result}")
        if set(decoder_result.missing_keys) != self.DECODER_MISSING or set(
            decoder_result.unexpected_keys
        ) != self.DECODER_UNEXPECTED:
            raise RuntimeError(f"Unexpected Cosmos decoder state mismatch: {decoder_result}")

        parameters = dict(self.named_parameters())
        for name in self.ENCODER_MISSING:
            parameters[f"encoder.{name}"].requires_grad_(False)
        for name in self.DECODER_MISSING:
            parameters[f"decoder.{name}"].requires_grad_(False)

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        return self.encoder(video)[0]

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)


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
