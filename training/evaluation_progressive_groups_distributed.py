from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import time
from typing import Any

from DISTS_pytorch import DISTS
import torch
import torch.nn.functional as F

from datasets.video_sequence import load_png_sequence
from entropy_model.model_categorical import CategoricalEntropyModel
from metrics.lpips import sequence_lpips
from tokenizer.stage1_vq import Stage1VQTokenizer
from tokenizer.stage3_vq import selected_code_vectors
from tokenizer.stage_backbones import TOKENIZER_SPECS
from training.evaluation_categorical import (
    DATASET_NAMES,
    _sequence_dists,
)
from training.evaluation_categorical_distributed import (
    SequenceJob,
    _expected_sequence_names,
    save_reconstruction,
    shard_jobs,
)
from training.uvg_protocol import (
    finish_uvg_reconstruction,
    prepare_uvg_input,
    uvg_metric_frames,
    uvg_source_frames,
)


PROGRESSIVE_LEVELS = (1, 2, 3, 4)


@dataclass(frozen=True)
class ProgressiveCompletion:
    indices: torch.Tensor
    selected: torch.Tensor
    retained_bits: torch.Tensor
    retained_indices: int
    generated_correct: int
    generated_indices: int


def complete_progressive_groups(
    entropy_model: CategoricalEntropyModel,
    codebook: torch.Tensor,
    true_indices: torch.Tensor,
    *,
    retained_groups: int,
) -> ProgressiveCompletion:
    """Keep the first K parity groups per frame and greedily fill the rest."""
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
    latent_frames = int(true_indices.shape[1])
    target_positions, source_positions = entropy_model._position_embeddings(
        latent_frames,
        true_indices.device,
        entropy_model.code_projection.weight.dtype,
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
        inputs = (
            inputs
            + target_positions[block_index].unsqueeze(0)
            + source_positions[block_index].unsqueeze(0)
        )
        hidden, cache = entropy_model.transformer.forward_step(inputs, cache)
        logits = entropy_model.output_head(entropy_model.output_norm(hidden))

        truth = true_blocks[:, block_index]
        block_valid = valid_blocks[:, block_index]
        group = block_index % layout.num_groups
        if group < retained_groups:
            chosen = truth
            log2_probabilities = torch.log_softmax(logits.float(), dim=-1) / math.log(2.0)
            symbol_bits = -log2_probabilities.gather(
                -1, truth.unsqueeze(-1)
            ).squeeze(-1)
            retained_bits = retained_bits + symbol_bits[block_valid].sum()
            retained_count = retained_count + block_valid.sum()
        else:
            chosen = logits.argmax(dim=-1)
            generated_correct = generated_correct + ((chosen == truth) & block_valid).sum()
            generated_count = generated_count + block_valid.sum()

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
def evaluate_progressive_sequence(
    job: SequenceJob,
    tokenizer: Stage1VQTokenizer,
    entropy_model: CategoricalEntropyModel,
    tokenizer_name: str,
    lpips_vgg_metric: torch.nn.Module,
    lpips_alex_metric: torch.nn.Module,
    dists_metric: DISTS,
    device: torch.device,
    recon_root: Path,
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
    rate_pixels = (
        prepared.rate_frames * reference.shape[-2] * reference.shape[-1]
    )
    reports = []
    for retained_groups in PROGRESSIVE_LEVELS:
        torch.cuda.synchronize(device)
        generation_started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            completion = complete_progressive_groups(
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
        save_reconstruction(
            reconstruction,
            recon_root / f"level_{retained_groups}" / job.dataset / job.sequence,
        )
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
                "latency_ms": (
                    analysis_ms + generation_ms + synthesis_ms
                ) / reference.shape[0],
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


def evaluate_progressive_shard(
    jobs: tuple[SequenceJob, ...],
    tokenizer: Stage1VQTokenizer,
    entropy_model: CategoricalEntropyModel,
    tokenizer_name: str,
    lpips_vgg_metric: torch.nn.Module,
    lpips_alex_metric: torch.nn.Module,
    dists_metric: DISTS,
    device: torch.device,
    recon_root: Path,
    *,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    reports = []
    for job in shard_jobs(jobs, rank, world_size):
        reports.extend(
            evaluate_progressive_sequence(
                job,
                tokenizer,
                entropy_model,
                tokenizer_name,
                lpips_vgg_metric,
                lpips_alex_metric,
                dists_metric,
                device,
                recon_root,
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
            "retained_groups": "first K parity groups in every latent frame",
            "missing_groups": "closed-loop per-site argmax generation",
            "rate": "estimated cross-entropy bits for retained indices only",
        },
        "sequences": reports,
    }


def _aggregate_progressive_dataset(
    dataset: str,
    retained_groups: int,
    reports: list[dict[str, Any]],
    tokenizer_name: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    total_frames = sum(int(report["frames"]) for report in reports)
    total_rate_pixels = sum(
        int(report["rate_frames"])
        * int(report["resolution"][0])
        * int(report["resolution"][1])
        for report in reports
    )
    total_error = sum(float(report["squared_error"]) for report in reports)
    total_values = sum(int(report["value_count"]) for report in reports)
    generated_indices = sum(int(report["generated_indices"]) for report in reports)
    sequences = [
        {name: value for name, value in report.items() if name != "dataset"}
        for report in reports
    ]
    return {
        "schema_version": 1,
        "dataset": DATASET_NAMES[dataset],
        "tokenizer": tokenizer_name,
        "retained_groups": retained_groups,
        "protocol": {
            "sequences": list(_expected_sequence_names(dataset)),
            **protocol,
        },
        "aggregate": {
            "psnr": -10.0 * math.log10(total_error / total_values),
            "bpp": sum(float(report["bits"]) for report in reports)
            / total_rate_pixels,
            "lpips": sum(
                float(report["lpips"]) * int(report["frames"])
                for report in reports
            ) / total_frames,
            "lpips_vgg": sum(
                float(report["lpips_vgg"]) * int(report["frames"])
                for report in reports
            ) / total_frames,
            "lpips_alex": sum(
                float(report["lpips_alex"]) * int(report["frames"])
                for report in reports
            ) / total_frames,
            "dists": sum(
                float(report["dists"]) * int(report["frames"])
                for report in reports
            ) / total_frames,
            "latency_ms": sum(
                float(report["latency_ms"]) * int(report["frames"])
                for report in reports
            ) / total_frames,
            "analysis_latency_ms": sum(
                float(report["analysis_latency_ms"]) * int(report["frames"])
                for report in reports
            ) / total_frames,
            "generation_latency_ms": sum(
                float(report["generation_latency_ms"]) * int(report["frames"])
                for report in reports
            ) / total_frames,
            "synthesis_latency_ms": sum(
                float(report["synthesis_latency_ms"]) * int(report["frames"])
                for report in reports
            ) / total_frames,
            "generated_top1_accuracy": (
                None
                if generated_indices == 0
                else sum(int(report["generated_correct"]) for report in reports)
                / generated_indices
            ),
        },
        "sequences": sequences,
    }


def merge_progressive_shards(
    shards: list[dict[str, Any]],
) -> dict[int, dict[str, dict[str, Any]]]:
    world_sizes = {int(shard["world_size"]) for shard in shards}
    if len(world_sizes) != 1:
        raise ValueError(f"Shards disagree on world_size: {world_sizes}")
    world_size = world_sizes.pop()
    ranks = {int(shard["rank"]) for shard in shards}
    if ranks != set(range(world_size)):
        raise ValueError(
            f"Expected shard ranks 0..{world_size - 1}, got {sorted(ranks)}"
        )
    tokenizers = {str(shard["tokenizer"]) for shard in shards}
    if len(tokenizers) != 1:
        raise ValueError(f"Shards disagree on tokenizer: {tokenizers}")
    tokenizer_name = tokenizers.pop()
    checkpoint_steps = {int(shard["checkpoint_step"]) for shard in shards}
    if len(checkpoint_steps) != 1:
        raise ValueError(f"Shards disagree on checkpoint step: {checkpoint_steps}")
    checkpoint_step = checkpoint_steps.pop()
    protocol = shards[0]["protocol"]
    if any(shard["protocol"] != protocol for shard in shards[1:]):
        raise ValueError("Shards disagree on protocol")

    by_identity: dict[tuple[str, str, int], dict[str, Any]] = {}
    for shard in shards:
        for report in shard["sequences"]:
            identity = (
                str(report["dataset"]),
                str(report["sequence"]),
                int(report["retained_groups"]),
            )
            if identity in by_identity:
                raise ValueError(f"Duplicate progressive evaluation: {identity}")
            by_identity[identity] = report
    expected = {
        (dataset, sequence, level)
        for dataset in ("uvg", "mcl_jcv", "hevc_b")
        for sequence in _expected_sequence_names(dataset)
        for level in PROGRESSIVE_LEVELS
    }
    if set(by_identity) != expected:
        missing = sorted(expected - set(by_identity))
        extra = sorted(set(by_identity) - expected)
        raise ValueError(f"Shard coverage changed: missing={missing}, extra={extra}")

    outputs: dict[int, dict[str, dict[str, Any]]] = {}
    for level in PROGRESSIVE_LEVELS:
        outputs[level] = {}
        for dataset in ("uvg", "mcl_jcv", "hevc_b"):
            reports = [
                by_identity[(dataset, sequence, level)]
                for sequence in _expected_sequence_names(dataset)
            ]
            output = _aggregate_progressive_dataset(
                dataset, level, reports, tokenizer_name, protocol
            )
            output["checkpoint_step"] = checkpoint_step
            outputs[level][dataset] = output
    return outputs
