from __future__ import annotations

import math
from types import MethodType

import torch


GAUSSIAN_VARIANCE = 0.01


def gaussian_tile_weight(
    height: int, width: int, device: torch.device
) -> torch.Tensor:
    sigma_y = height * math.sqrt(GAUSSIAN_VARIANCE)
    sigma_x = width * math.sqrt(GAUSSIAN_VARIANCE)
    y = torch.arange(height, device=device, dtype=torch.float32)
    x = torch.arange(width, device=device, dtype=torch.float32)
    y = torch.exp(-0.5 * ((y - (height - 1) / 2) / sigma_y).square())
    x = torch.exp(-0.5 * ((x - (width - 1) / 2) / sigma_x).square())
    return y[:, None] * x[None, :]


def gaussian_tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise RuntimeError("Gaussian tile evaluation requires one process per GPU")

    height = z.shape[-2] * self.vae_ratio
    width = z.shape[-1] * self.vae_ratio
    y_idx, y_len, _ = self.split_tiles(height, True)
    x_idx, x_len, _ = self.split_tiles(width, True)

    tiles = []
    placements = []
    for i_pos, i_len in zip(y_idx, y_len):
        latent_i_pos = i_pos // self.vae_ratio
        latent_i_len = i_len // self.vae_ratio
        for j_pos, j_len in zip(x_idx, x_len):
            latent_j_pos = j_pos // self.vae_ratio
            latent_j_len = j_len // self.vae_ratio
            tiles.append(
                z[
                    ...,
                    latent_i_pos : latent_i_pos + latent_i_len,
                    latent_j_pos : latent_j_pos + latent_j_len,
                ]
            )
            placements.append((i_pos, j_pos))

    tile_indices = list(range(len(tiles)))
    stack_tiling = self.stack_tiling and not (
        self.training and getattr(self.decoder, "mask_enabled", False)
    )
    decoded_tiles = self._run_tile_tasks(
        tiles, tile_indices, self.decode, stack_tiling
    )
    if len(decoded_tiles) != len(placements):
        raise RuntimeError(
            f"Decoded {len(decoded_tiles)} tiles for {len(placements)} placements"
        )

    first = decoded_tiles[0].to(z.device)
    accumulator = torch.zeros(
        (*first.shape[:-2], height, width), device=z.device, dtype=torch.float32
    )
    normalizer = torch.zeros(
        (1,) * (first.ndim - 2) + (height, width),
        device=z.device,
        dtype=torch.float32,
    )
    weights = {}
    for decoded, (i_pos, j_pos) in zip(decoded_tiles, placements):
        decoded = decoded.to(z.device)
        tile_height, tile_width = decoded.shape[-2:]
        if i_pos + tile_height > height or j_pos + tile_width > width:
            raise RuntimeError(
                f"Decoded tile at {(i_pos, j_pos)} with shape "
                f"{(tile_height, tile_width)} exceeds {(height, width)}"
            )
        shape = (tile_height, tile_width)
        if shape not in weights:
            weights[shape] = gaussian_tile_weight(
                tile_height, tile_width, z.device
            ).view((1,) * (decoded.ndim - 2) + shape)
        weight = weights[shape]
        output_slice = (
            ...,
            slice(i_pos, i_pos + tile_height),
            slice(j_pos, j_pos + tile_width),
        )
        accumulator[output_slice].add_(decoded.float() * weight)
        normalizer[output_slice].add_(weight)

    if float(normalizer.min()) <= 0.0:
        raise RuntimeError("Gaussian tile aggregation left uncovered output pixels")
    return (accumulator / normalizer).to(first.dtype)


def enable_gaussian_tile_decode(tokenizer: torch.nn.Module, tokenizer_name: str) -> None:
    if tokenizer_name != "minimax_h3":
        raise ValueError(
            f"Gaussian tile evaluation requires minimax_h3, got {tokenizer_name}"
        )
    vae = tokenizer.backbone.model
    vae.tiled_decode = MethodType(gaussian_tiled_decode, vae)


def gaussian_tile_blend_protocol() -> dict:
    return {
        "type": "gaussian",
        "gaussian_variance": GAUSSIAN_VARIANCE,
        "scope": "decoder_spatial_tiles",
    }
