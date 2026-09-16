from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .config_categorical import CategoricalEntropyModelConfig
from .config_local_global_memory import LocalGlobalMemoryEntropyConfig
from .layout import BlockTensor, SpatiotemporalWindowLayout
from .model_categorical import CategoricalEntropyModel


@dataclass(frozen=True)
class LocalGlobalMemoryEntropyOutput:
    logits: torch.Tensor
    estimated_bits: torch.Tensor
    bits_per_index: torch.Tensor


@dataclass(frozen=True)
class LocalGlobalMemoryState:
    memory: torch.Tensor
    local_history: torch.Tensor | None


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = x.chunk(2, dim=-1)
        return value * F.silu(gate)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_dim: int, *, self_attention: bool) -> None:
        super().__init__()
        self.self_attention = self_attention
        if self_attention:
            self.self_norm = nn.RMSNorm(dim)
            self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_norm = nn.RMSNorm(dim)
        self.context_norm = nn.RMSNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn_norm = nn.RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * ffn_dim, bias=False), SwiGLU(), nn.Linear(ffn_dim, dim, bias=False)
        )

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if self.self_attention:
            normalized = self.self_norm(queries)
            attended, _ = self.self_attn(normalized, normalized, normalized, need_weights=False)
            queries = queries + attended
        normalized_context = self.context_norm(context)
        attended, _ = self.cross_attn(
            self.cross_norm(queries), normalized_context, normalized_context, need_weights=False
        )
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class LocalGlobalMemoryEntropyModel(CategoricalEntropyModel):
    """Canonical same-window Stage2 model plus a 32-token FIFO-global readout."""

    def __init__(self, config: LocalGlobalMemoryEntropyConfig) -> None:
        super().__init__(CategoricalEntropyModelConfig(
            codebook_size=config.codebook_size, codebook_dim=config.codebook_dim,
            window_size=config.window_size, d_model=config.d_model,
            num_layers=config.num_layers, num_heads=config.num_heads,
            ffn_dim=config.ffn_dim, max_latent_frames=config.max_latent_frames,
        ))
        self.config = config
        dim = config.d_model
        self.frame_y = nn.Embedding(16, dim)
        self.frame_x = nn.Embedding(16, dim)
        self.initial_memory = nn.Parameter(torch.randn(config.memory_tokens, dim) * 0.02)
        self.summary_queries = nn.Parameter(torch.randn(config.summary_tokens, dim) * 0.02)
        self.summary_block = CrossAttentionBlock(dim, config.num_heads, config.ffn_dim, self_attention=True)
        self.summary_norm = nn.RMSNorm(dim)
        self.global_queries = nn.Parameter(torch.randn(config.global_tokens, dim) * 0.02)
        self.global_reader = nn.ModuleList(
            CrossAttentionBlock(dim, config.num_heads, config.ffn_dim, self_attention=True)
            for _ in range(config.global_read_layers)
        )
        self.global_norm = nn.RMSNorm(dim)
        self.global_fusion = CrossAttentionBlock(dim, config.num_heads, config.ffn_dim, self_attention=False)

    @staticmethod
    def _resize_1d_embedding(embedding: nn.Embedding, size: int) -> torch.Tensor:
        weight = embedding.weight.transpose(0, 1).unsqueeze(0)
        return F.interpolate(weight, size=size, mode="linear", align_corners=True)[0].transpose(0, 1)

    def _frame_vectors(self, quantized: torch.Tensor) -> torch.Tensor:
        projected = self.code_projection(quantized.permute(0, 2, 3, 4, 1).to(self.code_projection.weight.dtype))
        height, width = quantized.shape[-2:]
        y = self._resize_1d_embedding(self.frame_y, height)[:, None]
        x = self._resize_1d_embedding(self.frame_x, width)[None, :]
        return (projected + y + x).flatten(2, 3)

    def summarize_frames(self, frame_vectors: torch.Tensor) -> torch.Tensor:
        batch, frames, sites, dim = frame_vectors.shape
        context = frame_vectors.reshape(batch * frames, sites, dim)
        queries = self.summary_queries.unsqueeze(0).expand(batch * frames, -1, -1)
        summaries = self.summary_block(queries, context)
        return self.summary_norm(summaries).reshape(batch, frames, self.config.summary_tokens, dim)

    def memory_banks(self, summaries: torch.Tensor) -> torch.Tensor:
        batch, frames, summary_tokens, dim = summaries.shape
        initial = self.initial_memory.to(dtype=summaries.dtype).reshape(
            self.config.memory_frames, summary_tokens, dim
        ).unsqueeze(0).expand(batch, -1, -1, -1)
        history = torch.cat((initial, summaries), dim=1)
        starts = torch.arange(frames, device=summaries.device)[:, None]
        offsets = torch.arange(self.config.memory_frames, device=summaries.device)[None]
        return history[:, starts + offsets].reshape(batch, frames, self.config.memory_tokens, dim)

    def read_global(self, memory: torch.Tensor) -> torch.Tensor:
        queries = self.global_queries.unsqueeze(0).expand(memory.shape[0], -1, -1)
        for layer in self.global_reader:
            queries = layer(queries, memory)
        return self.global_norm(queries)

    def read_all_global(self, banks: torch.Tensor) -> torch.Tensor:
        batch, frames, tokens, dim = banks.shape
        result = self.read_global(banks.reshape(batch * frames, tokens, dim))
        return result.reshape(batch, frames, self.config.global_tokens, dim)

    def _fuse_global(
        self,
        hidden: torch.Tensor,
        blocks: BlockTensor,
        layout: SpatiotemporalWindowLayout,
        global_context: torch.Tensor,
    ) -> torch.Tensor:
        block_batch, ranks, sites, dim = hidden.shape
        time, _, _, _ = layout.coordinates(blocks.latent_frames, device=hidden.device)
        rank_time = time[:, 0]
        video_ids = torch.arange(blocks.batch_size, device=hidden.device).repeat_interleave(blocks.num_windows)
        contexts = global_context[video_ids[:, None], rank_time[None, :]]
        fused = self.global_fusion(
            hidden.reshape(block_batch * ranks, sites, dim),
            contexts.reshape(block_batch * ranks, self.config.global_tokens, dim),
        )
        return fused.reshape(block_batch, ranks, sites, dim)

    def predict(self, quantized: torch.Tensor, *, k: int) -> tuple[torch.Tensor, BlockTensor]:
        layout = self.layout(k)
        grid = quantized.permute(0, 2, 3, 4, 1).to(self.code_projection.weight.dtype)
        blocks = layout.to_blocks(grid)
        hidden = self.transformer(self._shifted_inputs(blocks, layout), layout.sites_per_block)
        frame_vectors = self._frame_vectors(quantized)
        summaries = self.summarize_frames(frame_vectors)
        global_context = self.read_all_global(self.memory_banks(summaries))
        hidden = self._fuse_global(hidden, blocks, layout, global_context)
        return self.output_head(self.output_norm(hidden)), blocks

    def forward(self, quantized: torch.Tensor, indices: torch.Tensor, *, k: int = 1) -> LocalGlobalMemoryEntropyOutput:
        self._validate(quantized, indices)
        logits, blocks = self.predict(quantized, k=k)
        targets = self._targets(indices, k=k)
        bits = F.cross_entropy(logits.float().reshape(-1, self.config.codebook_size), targets.reshape(-1), reduction="none").reshape(targets.shape) / math.log(2.0)
        estimated_bits = bits.sum(dim=(-1, -2)).reshape(indices.shape[0], blocks.num_windows).sum(dim=1)
        return LocalGlobalMemoryEntropyOutput(logits=logits, estimated_bits=estimated_bits, bits_per_index=estimated_bits.sum() / indices.numel())

    def init_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> LocalGlobalMemoryState:
        memory = self.initial_memory.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()
        return LocalGlobalMemoryState(memory=memory, local_history=None)

    def predict_rank(self, state: LocalGlobalMemoryState, decoded_frame: torch.Tensor, *, rank: int) -> torch.Tensor:
        if rank not in range(4):
            raise ValueError(f"rank must be in [0,4), got {rank}")
        frame = decoded_frame.unsqueeze(2)
        sequence = frame if state.local_history is None else torch.cat((state.local_history, frame), dim=2)
        logits, blocks = self.predict(sequence, k=1)
        current_rank = (sequence.shape[2] - 1) * 4 + rank
        return logits[:, current_rank].reshape(sequence.shape[0], blocks.num_windows, logits.shape[2], logits.shape[3])

    def update_state(self, state: LocalGlobalMemoryState, decoded_frame: torch.Tensor) -> LocalGlobalMemoryState:
        vectors = self._frame_vectors(decoded_frame.unsqueeze(2))
        summary = self.summarize_frames(vectors)[:, 0]
        memory = torch.cat((state.memory[:, self.config.summary_tokens:], summary), dim=1)
        frame = decoded_frame.unsqueeze(2)
        history = frame if state.local_history is None else torch.cat((state.local_history, frame), dim=2)
        return LocalGlobalMemoryState(memory=memory, local_history=history[:, :, -self.config.max_latent_frames:])


__all__ = ["LocalGlobalMemoryEntropyModel", "LocalGlobalMemoryEntropyOutput", "LocalGlobalMemoryState"]
