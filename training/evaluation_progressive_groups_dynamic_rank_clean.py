from __future__ import annotations

from dataclasses import replace
import math
import time
from typing import Any

from DISTS_pytorch import DISTS
import torch
import torch.nn.functional as F

from datasets.video_sequence import load_png_sequence
from entropy_model.model_categorical_dynamic_rank_reliable import (
    CategoricalEntropyModelDynamicRankReliable,
)
from metrics.lpips import sequence_lpips
from tokenizer.stage1_vq import Stage1VQTokenizer
from tokenizer.stage3_vq import selected_code_vectors
from tokenizer.stage_backbones import TOKENIZER_SPECS
from training.evaluation_categorical import _sequence_dists
from training.evaluation_categorical_distributed import SequenceJob, shard_jobs
from training.evaluation_progressive_groups_distributed import (
    PROGRESSIVE_LEVELS,
    ProgressiveCompletion,
)
from training.uvg_protocol import (
    finish_uvg_reconstruction,
    prepare_uvg_input,
    uvg_metric_frames,
    uvg_source_frames,
)


def complete_progressive_groups_dynamic_rank_clean(
    entropy_model: CategoricalEntropyModelDynamicRankReliable,
    codebook: torch.Tensor,
    true_indices: torch.Tensor,
    *,
    retained_groups: int,
) -> ProgressiveCompletion:
    """Keep the first K dynamic ranks per frame and greedily fill the rest."""
    if retained_groups not in PROGRESSIVE_LEVELS:
        raise ValueError(
            f"retained_groups must be one of {PROGRESSIVE_LEVELS}, "
            f"got {retained_groups}"
        )
    if true_indices.ndim != 4 or true_indices.dtype != torch.int64:
        raise ValueError(
            f"Expected int64 indices [B,T,H,W], got "
            f"{true_indices.dtype} {tuple(true_indices.shape)}"
        )

    window = entropy_model.config.window_size
    original_height, original_width = true_indices.shape[-2:]
    pad_height = (-original_height) % window
    pad_width = (-original_width) % window
    valid = torch.ones_like(true_indices, dtype=torch.bool)
    if pad_height or pad_width:
        true_indices = F.pad(true_indices, (0, pad_width, 0, pad_height))
        valid = F.pad(valid, (0, pad_width, 0, pad_height), value=False)

    layout = entropy_model.layout
    index_blocks = layout.to_blocks(true_indices.unsqueeze(-1))
    valid_blocks = layout.to_blocks(valid.unsqueeze(-1)).values.squeeze(-1)
    true_blocks = index_blocks.values.squeeze(-1)
    target_positions, source_positions = entropy_model._position_embeddings_for_blocks(
        index_blocks, entropy_model.code_projection.weight.dtype
    )

    cache = None
    previous_vectors = None
    chosen_blocks = []
    retained_bits = torch.zeros((), device=true_indices.device)
    retained_count = torch.zeros((), device=true_indices.device, dtype=torch.long)
    generated_correct = torch.zeros((), device=true_indices.device, dtype=torch.long)
    generated_count = torch.zeros((), device=true_indices.device, dtype=torch.long)

    for block_index in range(true_blocks.shape[1]):
        if block_index == 0:
            inputs = entropy_model.bos.expand(true_blocks.shape[0], -1, -1)
        else:
            inputs = entropy_model.code_projection(previous_vectors)
            status = torch.zeros(
                previous_vectors.shape[:2],
                dtype=torch.long,
                device=true_indices.device,
            )
            inputs = inputs + entropy_model.source_status_embedding(status)
        inputs = (
            inputs
            + target_positions[:, block_index]
            + source_positions[:, block_index]
        )
        hidden, cache = entropy_model.transformer.forward_step(inputs, cache)
        logits = entropy_model.output_head(entropy_model.output_norm(hidden))

        truth = true_blocks[:, block_index]
        block_valid = valid_blocks[:, block_index]
        generated = block_index % layout.num_groups >= retained_groups
        if generated:
            chosen = logits.argmax(dim=-1)
            generated_correct += ((chosen == truth) & block_valid).sum()
            generated_count += block_valid.sum()
        else:
            chosen = truth
            log2_probabilities = torch.log_softmax(logits.float(), dim=-1) / math.log(2.0)
            symbol_bits = -log2_probabilities.gather(-1, truth.unsqueeze(-1)).squeeze(-1)
            retained_bits += symbol_bits[block_valid].sum()
            retained_count += block_valid.sum()

        previous_vectors = F.embedding(chosen, codebook)
        previous_vectors = previous_vectors * block_valid.unsqueeze(-1)
        chosen_blocks.append(chosen)

    completed_blocks = replace(
        index_blocks,
        values=torch.stack(chosen_blocks, dim=1).unsqueeze(-1),
    )
    completed = layout.from_blocks(completed_blocks).squeeze(-1)
    completed = completed[..., :original_height, :original_width]
    return ProgressiveCompletion(
        indices=completed,
        selected=selected_code_vectors(codebook, completed),
        retained_bits=retained_bits,
        retained_indices=int(retained_count.item()),
        generated_correct=int(generated_correct.item()),
        generated_indices=int(generated_count.item()),
    )


@torch.inference_mode()
def evaluate_progressive_sequence_dynamic_rank_clean(
    job: SequenceJob,
    tokenizer: Stage1VQTokenizer,
    entropy_model: CategoricalEntropyModelDynamicRankReliable,
    tokenizer_name: str,
    lpips_vgg_metric: torch.nn.Module,
    lpips_alex_metric: torch.nn.Module,
    dists_metric: DISTS,
    device: torch.device,
) -> list[dict[str, Any]]:
    source_frames = uvg_source_frames(tokenizer_name)
    video, _ = load_png_sequence(job.frames_dir, source_frames, start_frame=1)
    prepared = prepare_uvg_input(video.float().div(255.0), tokenizer_name)

    torch.cuda.synchronize(device)
    analysis_started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        latent = tokenizer.encode(prepared.model_input.to(device))
        projected = tokenizer.pre_quant(latent)
    flat = projected.permute(0, 2, 3, 4, 1).reshape(-1, projected.shape[1]).float()
    flat_indices = tokenizer.quantizer._nearest_indices(flat)
    true_indices = flat_indices.reshape(
        projected.shape[0], projected.shape[2], projected.shape[3], projected.shape[4]
    )
    torch.cuda.synchronize(device)
    analysis_ms = (time.perf_counter() - analysis_started) * 1000.0

    reference = prepared.reference
    rate_pixels = prepared.rate_frames * reference.shape[-2] * reference.shape[-1]
    reports = []
    for retained_groups in PROGRESSIVE_LEVELS:
        torch.cuda.synchronize(device)
        generation_started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            completion = complete_progressive_groups_dynamic_rank_clean(
                entropy_model,
                tokenizer.quantizer.embed,
                true_indices,
                retained_groups=retained_groups,
            )
        torch.cuda.synchronize(device)
        generation_ms = (time.perf_counter() - generation_started) * 1000.0

        synthesis_started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reconstruction = tokenizer.decode(
                tokenizer.post_quant(completion.selected.to(projected.dtype))
            )
        torch.cuda.synchronize(device)
        synthesis_ms = (time.perf_counter() - synthesis_started) * 1000.0

        reconstruction = finish_uvg_reconstruction(reconstruction, prepared)
        difference = reference - reconstruction
        squared_error = float(difference.square().sum().item())
        value_count = difference.numel()
        lpips_vgg = sequence_lpips(
            reference, reconstruction, lpips_vgg_metric, device=device, batch_size=1
        )
        lpips_alex = sequence_lpips(
            reference, reconstruction, lpips_alex_metric, device=device, batch_size=1
        )
        dists = _sequence_dists(reference, reconstruction, dists_metric, device)
        generated_accuracy = (
            None
            if completion.generated_indices == 0
            else completion.generated_correct / completion.generated_indices
        )
        reports.append(
            {
                "dataset": job.dataset,
                "sequence": job.sequence,
                "retained_groups": retained_groups,
                "retained_group_fraction": retained_groups / 4.0,
                "frames": reference.shape[0],
                "rate_frames": prepared.rate_frames,
                "resolution": [reference.shape[-2], reference.shape[-1]],
                "psnr": -10.0 * math.log10(squared_error / value_count),
                "bpp": float(completion.retained_bits.item()) / rate_pixels,
                "lpips": lpips_vgg,
                "lpips_vgg": lpips_vgg,
                "lpips_alex": lpips_alex,
                "dists": dists,
                "latency_ms": (analysis_ms + generation_ms + synthesis_ms)
                / reference.shape[0],
                "analysis_latency_ms": analysis_ms / reference.shape[0],
                "generation_latency_ms": generation_ms / reference.shape[0],
                "synthesis_latency_ms": synthesis_ms / reference.shape[0],
                "bits": float(completion.retained_bits.item()),
                "retained_indices": completion.retained_indices,
                "generated_correct": completion.generated_correct,
                "generated_indices": completion.generated_indices,
                "generated_top1_accuracy": generated_accuracy,
                "squared_error": squared_error,
                "value_count": value_count,
            }
        )
        del completion, reconstruction, difference
    return reports


def evaluate_progressive_shard_dynamic_rank_clean(
    jobs: tuple[SequenceJob, ...],
    tokenizer: Stage1VQTokenizer,
    entropy_model: CategoricalEntropyModelDynamicRankReliable,
    tokenizer_name: str,
    lpips_vgg_metric: torch.nn.Module,
    lpips_alex_metric: torch.nn.Module,
    dists_metric: DISTS,
    device: torch.device,
    *,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    reports = []
    for job in shard_jobs(jobs, rank, world_size):
        reports.extend(
            evaluate_progressive_sequence_dynamic_rank_clean(
                job,
                tokenizer,
                entropy_model,
                tokenizer_name,
                lpips_vgg_metric,
                lpips_alex_metric,
                dists_metric,
                device,
            )
        )
    return {
        "schema_version": 1,
        "rank": rank,
        "world_size": world_size,
        "tokenizer": tokenizer_name,
        "protocol": {
            "model_input_frames": TOKENIZER_SPECS[tokenizer_name].input_frames,
            "metric_frames": uvg_metric_frames(tokenizer_name),
            "bpp_denominator_frames": uvg_source_frames(tokenizer_name),
            "levels": list(PROGRESSIVE_LEVELS),
            "retained_groups": "first K dynamic ranks in every latent frame",
            "missing_groups": "closed-loop per-site argmax generation",
            "source_status": "always 0, matching clean training",
            "attention": "dense causal KV cache, matching clean training",
            "rate": "estimated cross-entropy bits for retained indices only",
            "reconstruction_storage": "disabled for Stage2 diagnostic",
        },
        "sequences": reports,
    }
