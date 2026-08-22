from types import SimpleNamespace

import pytest
import torch

from tokenizer.gaussian_tile_decode import (
    enable_gaussian_tile_decode,
    gaussian_tile_blend_protocol,
    gaussian_tiled_decode,
)


class IdentityTiledVAE:
    vae_ratio = 1
    stack_tiling = False
    training = False
    decoder = SimpleNamespace(mask_enabled=False)

    def split_tiles(self, length: int, decode: bool):
        assert length == 4
        assert decode
        return [0, 2], [3, 2], None

    def decode(self, tile: torch.Tensor) -> torch.Tensor:
        return tile

    def _run_tile_tasks(self, tiles, indices, task, stack_tiling):
        assert indices == list(range(len(tiles)))
        assert not stack_tiling
        return [task(tile) for tile in tiles]


def test_gaussian_tiled_decode_reconstructs_identity_tiles() -> None:
    latent = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    reconstruction = gaussian_tiled_decode(IdentityTiledVAE(), latent)
    torch.testing.assert_close(reconstruction, latent)


def test_gaussian_install() -> None:
    vae = IdentityTiledVAE()
    tokenizer = SimpleNamespace(backbone=SimpleNamespace(model=vae))
    enable_gaussian_tile_decode(tokenizer, "minimax_h3")
    assert vae.tiled_decode.__self__ is vae
    assert gaussian_tile_blend_protocol()["gaussian_variance"] == 0.01
    with pytest.raises(ValueError, match="requires minimax_h3"):
        enable_gaussian_tile_decode(tokenizer, "wanvae")
