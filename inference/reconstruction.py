"""Single-sequence inference adapted from the original H3VC evaluator."""
from pathlib import Path
from typing import Any
import math
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from entropy_model.model_categorical import CategoricalEntropyModel
from .frames import load_png_sequence
from .protocol import prepare_uvg_input, finish_uvg_reconstruction

def selected_code_vectors(
    codebook: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    selected = F.embedding(indices, codebook)
    return selected.permute(0, 4, 1, 2, 3).contiguous()

def _estimated_bits(
    entropy_model: CategoricalEntropyModel,
    features: torch.Tensor,
    indices: torch.Tensor,
    *,
    k: int,
) -> torch.Tensor:
    window = entropy_model.config.window_size
    pad_height = (-indices.shape[-2]) % window
    pad_width = (-indices.shape[-1]) % window
    valid = torch.ones_like(indices, dtype=torch.bool)
    if pad_height or pad_width:
        features = F.pad(features, (0, pad_width, 0, pad_height))
        indices = F.pad(indices, (0, pad_width, 0, pad_height))
        valid = F.pad(valid, (0, pad_width, 0, pad_height), value=False)
    logits, _ = entropy_model.predict(features, k=k)
    layout = entropy_model.layout(k)
    targets = layout.to_blocks(indices.unsqueeze(-1)).values.squeeze(-1)
    valid_blocks = layout.to_blocks(valid.unsqueeze(-1)).values.squeeze(-1)
    bits = F.cross_entropy(
        logits.float().reshape(-1, entropy_model.config.codebook_size),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape) / math.log(2.0)
    return bits[valid_blocks].sum()

def codebook_statistics(counts: torch.Tensor) -> dict[str, Any]:
    total = int(counts.sum().item())
    if total == 0:
        raise ValueError("Cannot measure codebook utilization without assignments")
    probabilities = counts[counts > 0].double() / total
    entropy_bits = float(-(probabilities * probabilities.log2()).sum().item())
    return {
        "total_assignments": total,
        "codebook_size": counts.numel(),
        "active_codes": int((counts > 0).sum().item()),
        "marginal_entropy_bits": entropy_bits,
        "perplexity": 2.0 ** entropy_bits,
        "top1_mass": float(probabilities.max().item()),
    }

def save_reconstruction(reconstruction: torch.Tensor, output_dir: Path) -> None:
    output_dir.mkdir(parents=True)
    frames = reconstruction.mul(255.0).round().to(torch.uint8)
    for frame_index, frame in enumerate(frames):
        pixels = frame.permute(1, 2, 0).numpy()
        Image.fromarray(np.ascontiguousarray(pixels)).save(
            output_dir / f"{frame_index + 1:06d}.png"
        )


@torch.inference_mode()
def reconstruct(frames_dir, tokenizer, entropy_model, device, recon_root):
    video = load_png_sequence(frames_dir)
    prepared = prepare_uvg_input(video.float().div(255.0))

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        latent = tokenizer.encode(prepared.model_input.to(device))
        projected = tokenizer.pre_quant(latent)
    flat = projected.permute(0, 2, 3, 4, 1).reshape(
        -1, projected.shape[1]
    ).float()
    flat_indices = tokenizer.quantizer._nearest_indices(flat)
    indices = flat_indices.reshape(
        projected.shape[0],
        projected.shape[2],
        projected.shape[3],
        projected.shape[4],
    )
    selected = selected_code_vectors(tokenizer.quantizer.embed, indices)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        bits = float(_estimated_bits(entropy_model, selected, indices, k=1).item())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        reconstruction = tokenizer.decode(
            tokenizer.post_quant(selected.to(projected.dtype))
        )
    torch.cuda.synchronize(device)
    total_ms = (time.perf_counter() - started) * 1000.0
    counts = torch.bincount(flat_indices, minlength=tokenizer.quantizer.embed.shape[0]).cpu()

    reconstruction = finish_uvg_reconstruction(reconstruction, prepared)
    if not math.isfinite(bits) or not torch.isfinite(reconstruction).all():
        raise RuntimeError("Non-finite entropy rate or reconstruction")
    save_reconstruction(reconstruction, recon_root)
    reference = prepared.reference
    difference = reference - reconstruction
    squared_error = float(difference.square().sum().item())
    value_count = difference.numel()
    rate_frames = 96
    rate_pixels = rate_frames * reference.shape[-2] * reference.shape[-1]
    return {
        "codebook": codebook_statistics(counts),
        "codebook_histogram": counts.tolist(),
        "sequence": frames_dir.name,
        "frames": reference.shape[0],
        "rate_frames": rate_frames,
        "resolution": [reference.shape[-2], reference.shape[-1]],
        "psnr": -10.0 * math.log10(squared_error / value_count),
        "bpp": bits / rate_pixels,
        "latency_ms": total_ms / reference.shape[0],
        "bits": bits,
        "squared_error": squared_error,
        "value_count": value_count,
    }
