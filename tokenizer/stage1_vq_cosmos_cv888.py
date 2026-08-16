from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import torch
import torch.nn as nn


class CosmosCV888Backbone(nn.Module):
    ENCODER_MISSING = {
        "encoder.down.1.downsample.conv1.conv3d.weight",
        "encoder.down.1.downsample.conv1.conv3d.bias",
        "encoder.down.1.downsample.conv2.conv3d.weight",
        "encoder.down.1.downsample.conv2.conv3d.bias",
        "encoder.down.1.downsample.conv3.conv3d.weight",
        "encoder.down.1.downsample.conv3.conv3d.bias",
    }
    ENCODER_UNEXPECTED = {
        "encoder.patcher3d.wavelets",
        "encoder.patcher3d._arange",
        "encoder.patcher3d.patch_size_buffer",
    }
    DECODER_MISSING = {
        "decoder.up.2.upsample.conv1.conv3d.weight",
        "decoder.up.2.upsample.conv1.conv3d.bias",
        "decoder.up.2.upsample.conv2.conv3d.weight",
        "decoder.up.2.upsample.conv2.conv3d.bias",
        "decoder.up.2.upsample.conv3.conv3d.weight",
        "decoder.up.2.upsample.conv3.conv3d.bias",
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
            temporal_compression=8,
            spatial_compression=8,
            channels_mult=[2, 4, 4],
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
