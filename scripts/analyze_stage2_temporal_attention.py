from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.openvid_mp4_sequence import TrainingSampleKey
from datasets.trainable_jsons_sequence_fast import (
    SeededFastTrainableJsonsVideoDataset,
)
from entropy_model.model_categorical_context_corruption import (
    CategoricalEntropyModelContextCorruption,
)
from metrics.reporting import write_json
from scripts.stage2_cosine_context_corruption import load_checkpoint
from tokenizer.stage3_vq import selected_code_vectors
from tokenizer.stage_backbones import TOKENIZER_SPECS
from tokenizer.stage_checkpoint import load_stage1_tokenizer
from training.checkpoint_config_fast import assert_fast_loader_resume_config
from training.config_stage2_context_corruption import (
    load_stage2_context_corruption_config,
    stage2_context_corruption_config_dict,
)
from training.stage_data_fast import DEFAULT_DATASET_ROOT, DEFAULT_METADATA
from training.temporal_attention_mask_ablation import evaluate_temporal_radii


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Stage2 temporal attention using the TRA oracle protocol"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/stage2_context_corruption_cosine.yaml"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--crop-seed", type=int, default=2026)
    parser.add_argument("--target-mass", type=float, default=0.9)
    parser.add_argument(
        "--temporal-radii", type=int, nargs="+", default=[1, 2, 4, 8, 16]
    )
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--wan-repo", default=str(PROJECT_ROOT / "Wan2.1"))
    parser.add_argument(
        "--wan-checkpoint",
        default=str(PROJECT_ROOT / "pretrained_ckpts/Wan2.1/Wan2.1_VAE.pth"),
    )
    parser.add_argument("--cosmos-repo", default=str(PROJECT_ROOT / "Cosmos-Tokenizer"))
    parser.add_argument(
        "--cosmos-encoder",
        default=str(
            PROJECT_ROOT / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV4x8x8/encoder.jit"
        ),
    )
    parser.add_argument(
        "--cosmos-decoder",
        default=str(
            PROJECT_ROOT / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV4x8x8/decoder.jit"
        ),
    )
    parser.add_argument(
        "--cosmos-cv888-encoder",
        default=str(
            PROJECT_ROOT / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV8x8x8/encoder.jit"
        ),
    )
    parser.add_argument(
        "--cosmos-cv888-decoder",
        default=str(
            PROJECT_ROOT / "pretrained_ckpts/Cosmos-0.1-Tokenizer-CV8x8x8/decoder.jit"
        ),
    )
    parser.add_argument(
        "--minimax-h3-model-dir",
        default=str(PROJECT_ROOT / "pretrained_ckpts/MiniMax-H3/FL2VA/video_vae"),
    )
    return parser.parse_args()


def oracle_retained_density(
    probabilities: torch.Tensor, target_mass: float
) -> torch.Tensor:
    if not 0.0 < target_mass < 1.0:
        raise ValueError(f"target_mass must be within (0,1), got {target_mass}")
    sorted_probabilities = probabilities.sort(dim=-1, descending=True).values
    retained_keys = (
        sorted_probabilities.cumsum(dim=-1) < target_mass
    ).sum(dim=-1) + 1
    return retained_keys.float() / probabilities.shape[-1]


def minimum_recent_frame_radius(
    frame_mass: torch.Tensor, target_mass: float
) -> torch.Tensor:
    if not 0.0 < target_mass < 1.0:
        raise ValueError(f"target_mass must be within (0,1), got {target_mass}")
    required_mass = target_mass * frame_mass.sum(dim=-1)
    recent_first = frame_mass.flip(dims=(-1,)).cumsum(dim=-1)
    return (recent_first < required_mass.unsqueeze(-1)).sum(dim=-1) + 1


def shifted_source_block(block_index: int, num_groups: int = 4) -> tuple[int, int]:
    if block_index <= 0:
        raise ValueError("Block zero is BOS and has no source code block")
    source_index = block_index - 1
    return divmod(source_index, num_groups)


def _quantize_video(tokenizer, video: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
        latent = tokenizer.encode(video.unsqueeze(0).to(device))
        projected = tokenizer.pre_quant(latent)
    flat = projected.permute(0, 2, 3, 4, 1).reshape(-1, projected.shape[1]).float()
    indices = tokenizer.quantizer._nearest_indices(flat).reshape(
        projected.shape[0], projected.shape[2], projected.shape[3], projected.shape[4]
    )
    selected = selected_code_vectors(tokenizer.quantizer.embed, indices)
    return selected, indices


def analyze_attention(
    model: CategoricalEntropyModelContextCorruption,
    selected: torch.Tensor,
    target_mass: float,
) -> tuple[dict, list[dict], np.ndarray, dict[int, list[float]], tuple[np.ndarray, np.ndarray]]:
    blocks = model.layout.to_blocks(selected.permute(0, 2, 3, 4, 1))
    latent_frames = blocks.latent_frames
    num_groups = model.layout.num_groups
    sites = model.layout.sites_per_block
    if latent_frames < 2:
        raise ValueError(f"Temporal attention analysis requires at least 2 latent frames, got {latent_frames}")

    hidden = model._shifted_inputs(blocks.values).reshape(
        blocks.values.shape[0], blocks.values.shape[1] * sites, model.config.d_model
    )
    mask = model.transformer.attention_mask(
        blocks.values.shape[1], sites, device=hidden.device, dtype=hidden.dtype
    )

    records: list[dict] = []
    global_mass_sum = np.zeros((latent_frames, latent_frames, num_groups), dtype=np.float64)
    global_mass_count = np.zeros_like(global_mass_sum)
    radii_by_layer: dict[int, list[float]] = {
        layer_index: [] for layer_index in range(len(model.transformer.layers))
    }
    entropy_values: list[np.ndarray] = []
    density_values: list[np.ndarray] = []
    summary_rows: list[dict] = []

    for layer_index, layer in enumerate(model.transformer.layers):
        normalized = layer.attention_norm(hidden)
        query, key, _ = layer.attention._qkv(normalized)
        layer_entropy: list[torch.Tensor] = []
        layer_density: list[torch.Tensor] = []
        layer_history_mass: list[torch.Tensor] = []
        layer_radius: list[torch.Tensor] = []

        for target_frame in range(1, latent_frames):
            target_block = target_frame * num_groups
            query_start = target_block * sites
            query_end = query_start + sites
            allowed_end = query_end
            query_rows = query[:, :, query_start:query_end].float()
            allowed_keys = key[:, :, :allowed_end].float()
            scores = query_rows @ allowed_keys.transpose(-2, -1)
            scores = scores / math.sqrt(layer.attention.head_dim)
            probabilities = torch.softmax(scores, dim=-1)

            entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=-1)
            density = oracle_retained_density(probabilities, target_mass)
            bos_mass = probabilities[..., :sites].sum(dim=-1)
            history_probabilities = probabilities[..., sites:]
            history_mass = history_probabilities.sum(dim=-1)
            source_block_mass = history_probabilities.reshape(
                *history_probabilities.shape[:-1], target_frame * num_groups, sites
            ).sum(dim=-1)
            normalized_source_mass = source_block_mass / history_mass.unsqueeze(-1)
            source_frame_mass = source_block_mass.reshape(
                *source_block_mass.shape[:-1], target_frame, num_groups
            ).sum(dim=-1)
            recent_radius = minimum_recent_frame_radius(source_frame_mass, target_mass)

            layer_entropy.append(entropy.cpu())
            layer_density.append(density.cpu())
            layer_history_mass.append(history_mass.cpu())
            layer_radius.append(recent_radius.cpu())
            radii_by_layer[layer_index].extend(
                recent_radius.float().reshape(-1).cpu().tolist()
            )
            entropy_values.append(entropy.reshape(-1).cpu().numpy())
            density_values.append(density.reshape(-1).cpu().numpy())

            for head in range(model.config.num_heads):
                head_absolute = source_block_mass[:, head].mean(dim=(0, 1)).cpu()
                head_normalized = normalized_source_mass[:, head].mean(dim=(0, 1)).cpu()
                for source_block in range(target_frame * num_groups):
                    source_frame, source_group = divmod(source_block, num_groups)
                    absolute_mass = float(head_absolute[source_block])
                    normalized_mass = float(head_normalized[source_block])
                    records.append(
                        {
                            "layer": layer_index,
                            "head": head,
                            "target_frame": target_frame,
                            "source_frame": source_frame,
                            "temporal_lag": target_frame - source_frame,
                            "source_group": source_group,
                            "attention_mass": absolute_mass,
                            "history_normalized_mass": normalized_mass,
                            "mean_probability_per_key": absolute_mass / sites,
                        }
                    )

            aggregate_normalized = normalized_source_mass.mean(dim=(0, 1, 2)).cpu()
            for source_block in range(target_frame * num_groups):
                source_frame, source_group = divmod(source_block, num_groups)
                global_mass_sum[target_frame, source_frame, source_group] += float(
                    aggregate_normalized[source_block]
                )
                global_mass_count[target_frame, source_frame, source_group] += 1.0

            summary_rows.append(
                {
                    "layer": layer_index,
                    "target_frame": target_frame,
                    "attention_entropy_nats": float(entropy.mean().cpu()),
                    "oracle_retained_density": float(density.mean().cpu()),
                    "history_attention_mass": float(history_mass.mean().cpu()),
                    "bos_attention_mass": float(bos_mass.mean().cpu()),
                    "recent_frame_radius_mean": float(recent_radius.float().mean().cpu()),
                }
            )

        layer_entropy_tensor = torch.cat([value.reshape(-1) for value in layer_entropy])
        layer_density_tensor = torch.cat([value.reshape(-1) for value in layer_density])
        layer_history_tensor = torch.cat([value.reshape(-1) for value in layer_history_mass])
        layer_radius_tensor = torch.cat([value.reshape(-1) for value in layer_radius]).float()
        summary_rows.append(
            {
                "layer": layer_index,
                "target_frame": "all",
                "attention_entropy_nats": float(layer_entropy_tensor.mean()),
                "oracle_retained_density": float(layer_density_tensor.mean()),
                "history_attention_mass": float(layer_history_tensor.mean()),
                "recent_frame_radius_mean": float(layer_radius_tensor.mean()),
                "recent_frame_radius_median": float(layer_radius_tensor.median()),
                "recent_frame_radius_p90": float(torch.quantile(layer_radius_tensor, 0.9)),
            }
        )

        hidden = layer(hidden, mask)

    global_mass = np.divide(
        global_mass_sum,
        global_mass_count,
        out=np.zeros_like(global_mass_sum),
        where=global_mass_count > 0,
    )
    all_entropy = np.concatenate(entropy_values)
    all_density = np.concatenate(density_values)
    fit = np.polyfit(all_entropy, np.log(all_density), deg=1)
    predicted = np.polyval(fit, all_entropy)
    residual = np.square(np.log(all_density) - predicted).sum()
    total = np.square(np.log(all_density) - np.log(all_density).mean()).sum()
    correlation = {
        "log_density_fit_slope": float(fit[0]),
        "log_density_fit_intercept": float(fit[1]),
        "log_density_r_squared": float(1.0 - residual / total),
        "samples": int(all_entropy.size),
    }
    metadata = {
        "latent_frames": latent_frames,
        "latent_height": blocks.height,
        "latent_width": blocks.width,
        "spatial_windows": blocks.num_windows,
        "groups": num_groups,
        "sites_per_group": sites,
        "correlation": correlation,
        "layer_target_summary": summary_rows,
    }
    return metadata, records, global_mass, radii_by_layer, (all_entropy, all_density)


def write_records(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_results(
    output_dir: Path,
    global_mass: np.ndarray,
    radii_by_layer: dict[int, list[float]],
    entropy_density: tuple[np.ndarray, np.ndarray],
) -> None:
    latent_frames, _, num_groups = global_mass.shape
    matrix = np.full((latent_frames - 1, latent_frames * num_groups), np.nan)
    for target_frame in range(1, latent_frames):
        matrix[target_frame - 1, : target_frame * num_groups] = global_mass[
            target_frame, :target_frame
        ].reshape(-1)
    fig, axis = plt.subplots(figsize=(18, 7))
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma")
    axis.set_xlabel("Historical source block (frame-major, g0-g3)")
    axis.set_ylabel("Target frame, group 0")
    axis.set_yticks(np.arange(latent_frames - 1))
    axis.set_yticklabels(np.arange(1, latent_frames))
    axis.set_xticks(np.arange(0, latent_frames * num_groups, num_groups))
    axis.set_xticklabels(np.arange(latent_frames))
    fig.colorbar(image, ax=axis, label="Attention mass normalized within history")
    fig.tight_layout()
    fig.savefig(output_dir / "temporal_group_attention.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    layers = sorted(radii_by_layer)
    values = [radii_by_layer[layer] for layer in layers]
    axis.boxplot(values, tick_labels=[str(layer) for layer in layers], showfliers=False)
    axis.set_xlabel("Transformer layer")
    axis.set_ylabel("Recent frames needed for target historical mass")
    fig.tight_layout()
    fig.savefig(output_dir / "recent_frame_radius_by_layer.png", dpi=180)
    plt.close(fig)

    entropy, density = entropy_density
    stride = max(1, entropy.size // 20000)
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(entropy[::stride], density[::stride], s=3, alpha=0.12)
    axis.set_yscale("log")
    axis.set_xlabel("Attention entropy (nats)")
    axis.set_ylabel("Oracle retained density")
    fig.tight_layout()
    fig.savefig(output_dir / "entropy_vs_retained_density.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.target_mass < 1.0:
        raise ValueError(f"target_mass must be within (0,1), got {args.target_mass}")
    if not args.temporal_radii or min(args.temporal_radii) <= 0:
        raise ValueError("temporal_radii must contain positive integers")
    if len(set(args.temporal_radii)) != len(args.temporal_radii):
        raise ValueError("temporal_radii must not contain duplicates")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This diagnostic requires a CUDA device for the video tokenizer")
    torch.cuda.set_device(device)

    config = load_stage2_context_corruption_config(args.config)
    checkpoint = load_checkpoint(args.checkpoint)
    assert_fast_loader_resume_config(
        checkpoint["config"],
        stage2_context_corruption_config_dict(config),
        stage="Stage2 context corruption attention diagnostic",
    )
    dataset = SeededFastTrainableJsonsVideoDataset(
        args.dataset_root,
        args.metadata,
        num_frames=TOKENIZER_SPECS[checkpoint["tokenizer"]].input_frames,
        rejection_log=None,
    )
    video = dataset[TrainingSampleKey(args.sample_index, args.crop_seed)]

    started = time.perf_counter()
    loaded = load_stage1_tokenizer(
        checkpoint["tokenizer_checkpoint"],
        wan_repo=args.wan_repo,
        wan_checkpoint=args.wan_checkpoint,
        cosmos_repo=args.cosmos_repo,
        cosmos_encoder=args.cosmos_encoder,
        cosmos_decoder=args.cosmos_decoder,
        cosmos_cv888_encoder=args.cosmos_cv888_encoder,
        cosmos_cv888_decoder=args.cosmos_cv888_decoder,
        minimax_h3_model_dir=args.minimax_h3_model_dir,
    )
    tokenizer = loaded.model.to(device).eval().requires_grad_(False)
    selected, indices = _quantize_video(tokenizer, video, device)
    tokenizer.to("cpu")
    del tokenizer
    torch.cuda.empty_cache()

    model = CategoricalEntropyModelContextCorruption(config.model).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
        metadata, records, global_mass, radii_by_layer, entropy_density = analyze_attention(
            model, selected, args.target_mass
        )
        radius_ablation = evaluate_temporal_radii(
            model,
            selected,
            indices,
            sorted(args.temporal_radii),
            rate_pixels=video.shape[1] * video.shape[2] * video.shape[3],
            device=device,
        )
    torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - started) * 1000.0

    write_records(output_dir / "temporal_group_attention.csv", records)
    plot_results(output_dir, global_mass, radii_by_layer, entropy_density)
    report = {
        "schema_version": 1,
        "experiment": "stage2_temporal_attention_diagnostic",
        "protocol": {
            "paper": "Token Radius Attention, arXiv:2608.02504",
            "attention": "dense softmax probability from the trained Q/K projections",
            "query": "group 0 of every latent frame t >= 1",
            "keys": "BOS plus all four shifted source groups from latent frames [0,t-1]",
            "target_attention_mass": args.target_mass,
            "window_execution": "all spatial 8x8 windows remain parallel batch items",
        },
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "step": int(checkpoint["step"]),
            "tokenizer_checkpoint": str(Path(checkpoint["tokenizer_checkpoint"]).resolve()),
        },
        "input": {
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "metadata": str(Path(args.metadata).resolve()),
            "sample_index": args.sample_index,
            "crop_seed": args.crop_seed,
            "video_shape": list(video.shape),
            "index_shape": list(indices.shape),
        },
        "latency_ms": latency_ms,
        "attention": metadata,
        "temporal_radius_ablation": radius_ablation,
        "artifacts": {
            "detailed_csv": "temporal_group_attention.csv",
            "temporal_heatmap": "temporal_group_attention.png",
            "recent_radius_plot": "recent_frame_radius_by_layer.png",
            "entropy_density_plot": "entropy_vs_retained_density.png",
        },
    }
    write_json(output_dir / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
