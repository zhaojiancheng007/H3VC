from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LayerKVCache:
    key: torch.Tensor
    value: torch.Tensor


class BlockCausalSelfAttention(nn.Module):
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

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        query, key, value = self._qkv(x)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
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
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        attended = attended.transpose(1, 2).reshape(x.shape)
        return self.output(attended), LayerKVCache(key, value)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.input = nn.Linear(d_model, 2 * hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.input(x).chunk(2, dim=-1)
        return self.output(value * F.silu(gate))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(d_model)
        self.attention = BlockCausalSelfAttention(d_model, num_heads)
        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), attention_mask)
        return x + self.ffn(self.ffn_norm(x))

    def forward_step(
        self, x: torch.Tensor, cache: LayerKVCache | None
    ) -> tuple[torch.Tensor, LayerKVCache]:
        attended, cache = self.attention.forward_step(self.attention_norm(x), cache)
        x = x + attended
        return x + self.ffn(self.ffn_norm(x)), cache


class BlockCausalTransformer(nn.Module):
    def __init__(self, d_model: int, num_layers: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, ffn_dim) for _ in range(num_layers)]
        )

    @staticmethod
    def attention_mask(
        num_blocks: int,
        sites_per_block: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if num_blocks <= 0 or sites_per_block <= 0:
            raise ValueError("num_blocks and sites_per_block must be positive")
        block_ids = torch.arange(num_blocks, device=device).repeat_interleave(sites_per_block)
        allowed = block_ids[:, None] >= block_ids[None, :]
        mask = torch.zeros(allowed.shape, device=device, dtype=dtype)
        mask.masked_fill_(~allowed, float("-inf"))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor, sites_per_block: int) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,S,M,D], got {tuple(x.shape)}")
        batch, num_blocks, block_sites, channels = x.shape
        if block_sites != sites_per_block:
            raise ValueError(f"Expected M={sites_per_block}, got {block_sites}")
        flat = x.reshape(batch, num_blocks * block_sites, channels)
        mask = self.attention_mask(
            num_blocks, block_sites, device=x.device, dtype=x.dtype
        )
        for layer in self.layers:
            flat = layer(flat, mask)
        return flat.reshape(batch, num_blocks, block_sites, channels)

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
