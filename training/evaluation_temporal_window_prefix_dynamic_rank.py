from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
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
from training.evaluation_categorical import DATASET_NAMES, _sequence_dists
from training.evaluation_categorical_distributed import (
    SequenceJob,
    _expected_sequence_names,
    shard_jobs,
)
from training.uvg_protocol import (
    finish_uvg_reconstruction,
    prepare_uvg_input,
    uvg_metric_frames,
    uvg_source_frames,
)


TEMPORAL_WINDOW_LEVELS = (31, 65, 100, 135)


@dataclass(frozen=True)
class TemporalWindowPrefixCompletion:
    indices: torch.Tensor
    selected: torch.Tensor
    symbol_bits: torch.Tensor
    side_bits: int
    retained_indices: int
    generated_correct: int
    generated_indices: int
    retained_windows: int
    total_windows: int
    retained_window_counts_by_position: tuple[int, ...]
    retained_change_score_sum: float
    retained_change_score_count: int
    generated_change_score_sum: float
    generated_change_score_count: int

    @property
    def total_bits(self) -> float:
        return float(self.symbol_bits.item()) + self.side_bits


def fixed_cardinality_mask_bits(num_windows: int, retained_windows: int) -> int:
    if num_windows <= 0:
        raise ValueError(f"num_windows must be positive, got {num_windows}")
    if retained_windows < 0 or retained_windows > num_windows:
        raise ValueError(
            f"retained_windows must be in [0,{num_windows}], got {retained_windows}"
        )
    cardinality = math.comb(num_windows, retained_windows)
    return 0 if cardinality == 1 else (cardinality - 1).bit_length()


def encode_window_mask(mask: torch.Tensor) -> int:
    if mask.ndim != 1 or mask.dtype != torch.bool:
        raise ValueError(f"Expected bool mask [W], got {mask.dtype} {tuple(mask.shape)}")
    selected = tuple(mask.nonzero(as_tuple=False).flatten().tolist())
    candidates = combinations(range(mask.numel()), len(selected))
    for rank, candidate in enumerate(candidates):
        if candidate == selected:
            return rank
    raise RuntimeError("Window mask was not found in its fixed-cardinality codebook")


def decode_window_mask(
    rank: int,
    *,
    num_windows: int,
    retained_windows: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    cardinality = math.comb(num_windows, retained_windows)
    if rank < 0 or rank >= cardinality:
        raise ValueError(f"rank must be in [0,{cardinality}), got {rank}")
    selected = next(
        candidate
        for index, candidate in enumerate(combinations(range(num_windows), retained_windows))
        if index == rank
    )
    mask = torch.zeros(num_windows, dtype=torch.bool, device=device)
    mask[list(selected)] = True
    return mask


def temporal_window_change_scores(
    current: torch.Tensor,
    reference: torch.Tensor,
    codebook: torch.Tensor,
    *,
    window_size: int,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    if current.ndim != 3 or current.dtype != torch.int64:
        raise ValueError(
            f"Expected int64 current indices [B,H,W], got "
            f"{current.dtype} {tuple(current.shape)}"
        )
    if reference.shape != current.shape or reference.dtype != torch.int64:
        raise ValueError("reference must match current int64 indices")
    batch, height, width = map(int, current.shape)
    if height % window_size or width % window_size:
        raise ValueError(
            f"Latent size {(height, width)} must be divisible by window_size={window_size}"
        )
    if valid is None:
        valid = torch.ones_like(current, dtype=torch.bool)
    if valid.shape != current.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be a bool tensor matching current")

    squared_delta = (
        F.embedding(current, codebook).float()
        - F.embedding(reference, codebook).float()
    ).square().mean(dim=-1)
    windows_y = height // window_size
    windows_x = width // window_size
    squared_delta = squared_delta.reshape(
        batch, windows_y, window_size, windows_x, window_size
    ).permute(0, 1, 3, 2, 4)
    valid_windows = valid.reshape(
        batch, windows_y, window_size, windows_x, window_size
    ).permute(0, 1, 3, 2, 4)
    score_sum = (squared_delta * valid_windows).sum(dim=(-1, -2))
    valid_count = valid_windows.sum(dim=(-1, -2))
    if (valid_count == 0).any():
        raise ValueError("Every temporal-prefix window must contain a valid latent site")
    return (score_sum / valid_count).reshape(batch, windows_y * windows_x)


def select_dynamic_windows(scores: torch.Tensor, retained_windows: int) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError(f"Expected scores [B,W], got {tuple(scores.shape)}")
    num_windows = int(scores.shape[1])
    if retained_windows < 0 or retained_windows > num_windows:
        raise ValueError(
            f"retained_windows must be in [0,{num_windows}], got {retained_windows}"
        )
    selected = torch.zeros_like(scores, dtype=torch.bool)
    if retained_windows:
        indices = scores.topk(retained_windows, dim=1, largest=True, sorted=True).indices
        selected.scatter_(1, indices, True)
    return selected


def complete_temporal_window_prefix_dynamic_rank(
    entropy_model: CategoricalEntropyModelDynamicRankReliable,
    codebook: torch.Tensor,
    true_indices: torch.Tensor,
    *,
    retained_windows: int,
) -> TemporalWindowPrefixCompletion:
    """Transmit temporally novel windows and generate stable windows."""
    if true_indices.ndim != 4 or true_indices.dtype != torch.int64:
        raise ValueError(
            f"Expected int64 indices [B,T,H,W], got "
            f"{true_indices.dtype} {tuple(true_indices.shape)}"
        )

    window_size = entropy_model.config.window_size
    original_height, original_width = true_indices.shape[-2:]
    pad_height = (-original_height) % window_size
    pad_width = (-original_width) % window_size
    valid = torch.ones_like(true_indices, dtype=torch.bool)
    if pad_height or pad_width:
        true_indices = F.pad(true_indices, (0, pad_width, 0, pad_height))
        valid = F.pad(valid, (0, pad_width, 0, pad_height), value=False)

    layout = entropy_model.layout
    index_blocks = layout.to_blocks(true_indices.unsqueeze(-1))
    valid_blocks = layout.to_blocks(valid.unsqueeze(-1)).values.squeeze(-1)
    true_blocks = index_blocks.values.squeeze(-1)
    num_windows = index_blocks.num_windows
    if retained_windows < 0 or retained_windows > num_windows:
        raise ValueError(
            f"retained_windows must be in [0,{num_windows}], got {retained_windows}"
        )
    target_positions, source_positions = entropy_model._position_embeddings_for_blocks(
        index_blocks, entropy_model.code_projection.weight.dtype
    )

    completed_values = torch.zeros_like(true_blocks)
    cache = None
    previous_vectors = None
    symbol_bits = torch.zeros((), device=true_indices.device)
    retained_indices = 0
    generated_correct = 0
    generated_indices = 0
    retained_window_count = 0
    retained_by_position = torch.zeros(
        num_windows, dtype=torch.long, device=true_indices.device
    )
    retained_score_sum = 0.0
    retained_score_count = 0
    generated_score_sum = 0.0
    generated_score_count = 0
    previous_completed_frame = None

    for frame_index in range(index_blocks.latent_frames):
        if frame_index == 0:
            frame_retained = torch.ones(
                index_blocks.batch_size,
                num_windows,
                dtype=torch.bool,
                device=true_indices.device,
            )
        else:
            scores = temporal_window_change_scores(
                true_indices[:, frame_index],
                previous_completed_frame,
                codebook,
                window_size=window_size,
                valid=valid[:, frame_index],
            )
            frame_retained = select_dynamic_windows(scores, retained_windows)
            retained_score_sum += float(scores[frame_retained].sum().item())
            retained_score_count += int(frame_retained.sum().item())
            frame_generated = ~frame_retained
            generated_score_sum += float(scores[frame_generated].sum().item())
            generated_score_count += int(frame_generated.sum().item())

        retained_window_count += int(frame_retained.sum().item())
        retained_by_position += frame_retained.sum(dim=0)
        flat_retained = frame_retained.reshape(-1)
        for rank in range(layout.num_groups):
            block_index = frame_index * layout.num_groups + rank
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
            retained_sites = block_valid & flat_retained[:, None]
            generated_sites = block_valid & ~flat_retained[:, None]
            predicted = logits.argmax(dim=-1)
            chosen = torch.where(flat_retained[:, None], truth, predicted)

            log2_probabilities = torch.log_softmax(logits.float(), dim=-1) / math.log(2.0)
            per_symbol_bits = -log2_probabilities.gather(
                -1, truth.unsqueeze(-1)
            ).squeeze(-1)
            symbol_bits += per_symbol_bits[retained_sites].sum()
            retained_indices += int(retained_sites.sum().item())
            generated_correct += int(((predicted == truth) & generated_sites).sum().item())
            generated_indices += int(generated_sites.sum().item())

            completed_values[:, block_index] = chosen
            previous_vectors = F.embedding(chosen, codebook)
            previous_vectors = previous_vectors * block_valid.unsqueeze(-1)

        completed_blocks = replace(index_blocks, values=completed_values.unsqueeze(-1))
        completed_grid = layout.from_blocks(completed_blocks).squeeze(-1)
        previous_completed_frame = completed_grid[:, frame_index]

    completed = completed_grid[..., :original_height, :original_width]
    side_bits_per_frame = fixed_cardinality_mask_bits(num_windows, retained_windows)
    side_bits = (
        index_blocks.batch_size
        * max(index_blocks.latent_frames - 1, 0)
        * side_bits_per_frame
    )
    return TemporalWindowPrefixCompletion(
        indices=completed,
        selected=selected_code_vectors(codebook, completed),
        symbol_bits=symbol_bits,
        side_bits=side_bits,
        retained_indices=retained_indices,
        generated_correct=generated_correct,
        generated_indices=generated_indices,
        retained_windows=retained_window_count,
        total_windows=(
            index_blocks.batch_size * index_blocks.latent_frames * num_windows
        ),
        retained_window_counts_by_position=tuple(retained_by_position.cpu().tolist()),
        retained_change_score_sum=retained_score_sum,
        retained_change_score_count=retained_score_count,
        generated_change_score_sum=generated_score_sum,
        generated_change_score_count=generated_score_count,
    )


@torch.inference_mode()
def evaluate_temporal_window_prefix_sequence(
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
    num_windows = (
        math.ceil(true_indices.shape[-2] / entropy_model.config.window_size)
        * math.ceil(true_indices.shape[-1] / entropy_model.config.window_size)
    )
    levels = tuple(level for level in TEMPORAL_WINDOW_LEVELS if level <= num_windows)
    for retained_windows in levels:
        torch.cuda.synchronize(device)
        generation_started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            completion = complete_temporal_window_prefix_dynamic_rank(
                entropy_model,
                tokenizer.quantizer.embed,
                true_indices,
                retained_windows=retained_windows,
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
                "retained_windows_per_predicted_frame": retained_windows,
                "window_count": num_windows,
                "retained_window_fraction": (
                    completion.retained_windows / completion.total_windows
                ),
                "frames": reference.shape[0],
                "rate_frames": prepared.rate_frames,
                "resolution": [reference.shape[-2], reference.shape[-1]],
                "psnr": -10.0 * math.log10(squared_error / value_count),
                "bpp": completion.total_bits / rate_pixels,
                "symbol_bpp": float(completion.symbol_bits.item()) / rate_pixels,
                "side_bpp": completion.side_bits / rate_pixels,
                "lpips": lpips_vgg,
                "lpips_vgg": lpips_vgg,
                "lpips_alex": lpips_alex,
                "dists": dists,
                "latency_ms": (analysis_ms + generation_ms + synthesis_ms)
                / reference.shape[0],
                "analysis_latency_ms": analysis_ms / reference.shape[0],
                "generation_latency_ms": generation_ms / reference.shape[0],
                "synthesis_latency_ms": synthesis_ms / reference.shape[0],
                "bits": completion.total_bits,
                "symbol_bits": float(completion.symbol_bits.item()),
                "side_bits": completion.side_bits,
                "retained_indices": completion.retained_indices,
                "generated_correct": completion.generated_correct,
                "generated_indices": completion.generated_indices,
                "generated_top1_accuracy": generated_accuracy,
                "retained_window_counts_by_position": list(
                    completion.retained_window_counts_by_position
                ),
                "mean_retained_change_score": (
                    None
                    if completion.retained_change_score_count == 0
                    else completion.retained_change_score_sum
                    / completion.retained_change_score_count
                ),
                "mean_generated_change_score": (
                    None
                    if completion.generated_change_score_count == 0
                    else completion.generated_change_score_sum
                    / completion.generated_change_score_count
                ),
                "squared_error": squared_error,
                "value_count": value_count,
            }
        )
        del completion, reconstruction, difference
    return reports


def evaluate_temporal_window_prefix_shard(
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
            evaluate_temporal_window_prefix_sequence(
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
            "levels": list(TEMPORAL_WINDOW_LEVELS),
            "anchor": "all windows in the first latent frame are transmitted",
            "change_score": (
                "mean codebook-embedding squared distance from the decoder's "
                "completed previous frame at the same window position"
            ),
            "retained_windows": "top-K change scores per later latent frame",
            "missing_windows": "closed-loop per-rank argmax generation",
            "mask_side_information": (
                "fixed-length lexicographic combination rank; "
                "ceil(log2(comb(window_count,K))) bits per later latent frame"
            ),
            "rate": "estimated categorical symbol bits plus mask side bits",
            "reconstruction_storage": "disabled for Stage2 diagnostic",
        },
        "sequences": reports,
    }


def _aggregate_temporal_window_dataset(
    dataset: str,
    retained_windows: int,
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
        "retained_windows_per_predicted_frame": retained_windows,
        "protocol": {
            "sequences": list(_expected_sequence_names(dataset)),
            **protocol,
        },
        "aggregate": {
            "psnr": -10.0 * math.log10(total_error / total_values),
            "bpp": sum(float(report["bits"]) for report in reports)
            / total_rate_pixels,
            "symbol_bpp": sum(float(report["symbol_bits"]) for report in reports)
            / total_rate_pixels,
            "side_bpp": sum(float(report["side_bits"]) for report in reports)
            / total_rate_pixels,
            "lpips": sum(
                float(report["lpips"]) * int(report["frames"])
                for report in reports
            )
            / total_frames,
            "lpips_vgg": sum(
                float(report["lpips_vgg"]) * int(report["frames"])
                for report in reports
            )
            / total_frames,
            "lpips_alex": sum(
                float(report["lpips_alex"]) * int(report["frames"])
                for report in reports
            )
            / total_frames,
            "dists": sum(
                float(report["dists"]) * int(report["frames"])
                for report in reports
            )
            / total_frames,
            "latency_ms": sum(
                float(report["latency_ms"]) * int(report["frames"])
                for report in reports
            )
            / total_frames,
            "analysis_latency_ms": sum(
                float(report["analysis_latency_ms"]) * int(report["frames"])
                for report in reports
            )
            / total_frames,
            "generation_latency_ms": sum(
                float(report["generation_latency_ms"]) * int(report["frames"])
                for report in reports
            )
            / total_frames,
            "synthesis_latency_ms": sum(
                float(report["synthesis_latency_ms"]) * int(report["frames"])
                for report in reports
            )
            / total_frames,
            "generated_top1_accuracy": (
                None
                if generated_indices == 0
                else sum(int(report["generated_correct"]) for report in reports)
                / generated_indices
            ),
        },
        "sequences": sequences,
    }


def merge_temporal_window_prefix_shards(
    shards: list[dict[str, Any]],
) -> dict[int, dict[str, dict[str, Any]]]:
    world_sizes = {int(shard["world_size"]) for shard in shards}
    if len(world_sizes) != 1:
        raise ValueError(f"Shards disagree on world_size: {world_sizes}")
    world_size = world_sizes.pop()
    ranks = {int(shard["rank"]) for shard in shards}
    if ranks != set(range(world_size)):
        raise ValueError(f"Expected shard ranks 0..{world_size - 1}, got {sorted(ranks)}")
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
                int(report["retained_windows_per_predicted_frame"]),
            )
            if identity in by_identity:
                raise ValueError(f"Duplicate temporal-window evaluation: {identity}")
            by_identity[identity] = report
    expected = {
        (dataset, sequence, level)
        for dataset in ("uvg", "mcl_jcv", "hevc_b")
        for sequence in _expected_sequence_names(dataset)
        for level in TEMPORAL_WINDOW_LEVELS
    }
    if set(by_identity) != expected:
        missing = sorted(expected - set(by_identity))
        extra = sorted(set(by_identity) - expected)
        raise ValueError(f"Shard coverage changed: missing={missing}, extra={extra}")

    outputs: dict[int, dict[str, dict[str, Any]]] = {}
    for level in TEMPORAL_WINDOW_LEVELS:
        outputs[level] = {}
        for dataset in ("uvg", "mcl_jcv", "hevc_b"):
            reports = [
                by_identity[(dataset, sequence, level)]
                for sequence in _expected_sequence_names(dataset)
            ]
            output = _aggregate_temporal_window_dataset(
                dataset, level, reports, tokenizer_name, protocol
            )
            output["checkpoint_step"] = checkpoint_step
            outputs[level][dataset] = output
    return outputs
