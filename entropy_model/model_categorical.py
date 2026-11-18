from __future__ import annotations


import torch
from torch import nn

from .config_categorical import CategoricalEntropyModelConfig
from .layout import BlockTensor, SpatiotemporalWindowLayout
from .transformer import BlockCausalTransformer


class CategoricalEntropyModel(nn.Module):
    """One shared 3D-window entropy model with an explicit temporal length K."""

    def __init__(self, config: CategoricalEntropyModelConfig) -> None:
        super().__init__()
        self.config = config
        sites = config.window_size**2 // 4
        self.code_projection = nn.Linear(config.codebook_dim, config.d_model)
        self.bos = nn.Parameter(torch.zeros(1, sites, config.d_model))
        self.target_time = nn.Embedding(config.max_latent_frames, config.d_model)
        self.target_group = nn.Embedding(4, config.d_model)
        self.target_y = nn.Embedding(config.window_size, config.d_model)
        self.target_x = nn.Embedding(config.window_size, config.d_model)
        self.source_time = nn.Embedding(config.max_latent_frames, config.d_model)
        self.source_group = nn.Embedding(4, config.d_model)
        self.source_y = nn.Embedding(config.window_size, config.d_model)
        self.source_x = nn.Embedding(config.window_size, config.d_model)
        self.transformer = BlockCausalTransformer(
            config.d_model,
            config.num_layers,
            config.num_heads,
            config.ffn_dim,
        )
        self.output_norm = nn.RMSNorm(config.d_model)
        self.output_head = nn.Linear(
            config.d_model, config.codebook_size, bias=False
        )
        nn.init.zeros_(self.output_head.weight)

    def layout(self, k: int) -> SpatiotemporalWindowLayout:
        return SpatiotemporalWindowLayout(self.config.window_size, k)


    def _position_embeddings(
        self,
        blocks: BlockTensor,
        layout: SpatiotemporalWindowLayout,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time, group, local_y, local_x = layout.coordinates(
            blocks.latent_frames, device=self.bos.device
        )
        target = (
            self.target_time(time)
            + self.target_group(group)
            + self.target_y(local_y)
            + self.target_x(local_x)
        )
        source_time = time.roll(1, dims=0)
        source_group = group.roll(1, dims=0)
        source_y = local_y.roll(1, dims=0)
        source_x = local_x.roll(1, dims=0)
        source = (
            self.source_time(source_time)
            + self.source_group(source_group)
            + self.source_y(source_y)
            + self.source_x(source_x)
        )
        source[0] = 0
        return target.to(dtype), source.to(dtype)

    def _shifted_inputs(
        self, blocks: BlockTensor, layout: SpatiotemporalWindowLayout
    ) -> torch.Tensor:
        projected = self.code_projection(blocks.values)
        inputs = torch.empty_like(projected)
        inputs[:, 0] = self.bos.expand(projected.shape[0], -1, -1)
        inputs[:, 1:] = projected[:, :-1]
        target, source = self._position_embeddings(blocks, layout, projected.dtype)
        return inputs + target.unsqueeze(0) + source.unsqueeze(0)


    def predict(
        self, quantized: torch.Tensor, *, k: int
    ) -> tuple[torch.Tensor, BlockTensor]:
        layout = self.layout(k)
        grid = quantized.permute(0, 2, 3, 4, 1).to(
            self.code_projection.weight.dtype
        )
        blocks = layout.to_blocks(grid)
        hidden = self.transformer(
            self._shifted_inputs(blocks, layout), layout.sites_per_block
        )
        return self.output_head(self.output_norm(hidden)), blocks
