from __future__ import annotations

from functools import lru_cache

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from .transformer import LayerKVCache, SwiGLU


compiled_flex_attention = torch.compile(flex_attention, dynamic=False)


@lru_cache(maxsize=None)
def row_group_block_mask(
    num_stages: int,
    groups_per_frame: int,
    sites_per_group: int,
    device: torch.device,
):
    sequence_length = num_stages * sites_per_group

    def mask_mod(batch, head, query_index, key_index):
        del batch, head
        query_stage = query_index // sites_per_group
        key_stage = key_index // sites_per_group
        query_frame = query_stage // groups_per_frame
        first_key_stage = torch.where(
            query_frame == 0,
            0,
            (query_frame - 1) * groups_per_frame + 1,
        )
        return (key_stage >= first_key_stage) & (key_stage <= query_stage)

    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=device,
        _compile=True,
    )


class RowGroupSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

    def _qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        return qkv.unbind(dim=0)

    def forward(self, x: torch.Tensor, block_mask) -> torch.Tensor:
        query, key, value = self._qkv(x)
        attended = compiled_flex_attention(query, key, value, block_mask=block_mask)
        attended = attended.transpose(1, 2).reshape(x.shape)
        return self.output(attended)

    def forward_dense(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        query, key, value = self._qkv(x)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(x.shape)
        return self.output(attended)

    def forward_step(
        self, x: torch.Tensor, cache: LayerKVCache | None
    ) -> tuple[torch.Tensor, LayerKVCache]:
        query, current_key, current_value = self._qkv(x)
        if cache is None:
            key, value = current_key, current_value
        else:
            key = torch.cat((cache.key, current_key), dim=2)
            value = torch.cat((cache.value, current_value), dim=2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(x.shape)
        return self.output(attended), LayerKVCache(key, value)


class RowGroupTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(d_model)
        self.attention = RowGroupSelfAttention(d_model, num_heads)
        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_dim)

    def forward(self, x: torch.Tensor, block_mask) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), block_mask)
        return x + self.ffn(self.ffn_norm(x))

    def forward_dense(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attention.forward_dense(self.attention_norm(x), attention_mask)
        return x + self.ffn(self.ffn_norm(x))

    def forward_step(
        self, x: torch.Tensor, cache: LayerKVCache | None
    ) -> tuple[torch.Tensor, LayerKVCache]:
        attended, cache = self.attention.forward_step(self.attention_norm(x), cache)
        x = x + attended
        return x + self.ffn(self.ffn_norm(x)), cache


class RowGroupBlockCausalTransformer(nn.Module):
    def __init__(self, d_model: int, num_layers: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [RowGroupTransformerBlock(d_model, num_heads, ffn_dim) for _ in range(num_layers)]
        )

    @staticmethod
    def dense_attention_mask(
        num_stages: int,
        groups_per_frame: int,
        sites_per_group: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        sequence_length = num_stages * sites_per_group
        token_ids = torch.arange(sequence_length, device=device)
        query_stage = token_ids[:, None] // sites_per_group
        key_stage = token_ids[None, :] // sites_per_group
        query_frame = query_stage // groups_per_frame
        first_key_stage = torch.where(
            query_frame == 0,
            0,
            (query_frame - 1) * groups_per_frame + 1,
        )
        allowed = (key_stage >= first_key_stage) & (key_stage <= query_stage)
        return allowed.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        *,
        groups_per_frame: int,
        sites_per_group: int,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,S,M,D], got {tuple(x.shape)}")
        batch, num_stages, block_sites, channels = x.shape
        if block_sites != sites_per_group:
            raise ValueError(f"Expected M={sites_per_group}, got {block_sites}")
        flat = x.reshape(batch, num_stages * block_sites, channels)

        if x.device.type == "cuda":
            attention_mask = row_group_block_mask(
                num_stages,
                groups_per_frame,
                sites_per_group,
                x.device,
            )
            for layer in self.layers:
                flat = layer(flat, attention_mask)
        else:
            attention_mask = self.dense_attention_mask(
                num_stages,
                groups_per_frame,
                sites_per_group,
                device=x.device,
            )
            for layer in self.layers:
                flat = layer.forward_dense(flat, attention_mask)
        return flat.reshape(batch, num_stages, block_sites, channels)

    def forward_step(
        self,
        x: torch.Tensor,
        cache: tuple[LayerKVCache | None, ...] | None,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        if x.ndim != 3:
            raise ValueError(f"Expected current block [B,M,D], got {tuple(x.shape)}")
        if cache is None:
            layer_caches: tuple[LayerKVCache | None, ...] = (None,) * len(self.layers)
        else:
            if len(cache) != len(self.layers):
                raise ValueError(f"Expected {len(self.layers)} layer caches, got {len(cache)}")
            layer_caches = cache
        updated: list[LayerKVCache] = []
        for layer, layer_cache in zip(self.layers, layer_caches, strict=True):
            x, new_cache = layer.forward_step(x, layer_cache)
            updated.append(new_cache)
        return x, tuple(updated)
