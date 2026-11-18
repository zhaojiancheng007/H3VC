"""Single-sequence inference adapted from the original H3VC evaluator."""
from pathlib import Path
import math
import numpy as np
import lpips
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

def entropy_bits(
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

@torch.inference_mode()
def sequence_lpips(reference, reconstruction, metric, device):
    """Mean frame-wise LPIPS-VGG on RGB tensors in [0, 1]."""
    total = 0.0
    for ref, rec in zip(reference, reconstruction):
        total += metric(ref[None].to(device), rec[None].to(device), normalize=True).item()
    return total / reference.shape[0]


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
        bits = float(entropy_bits(entropy_model, selected, indices, k=1).item())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        reconstruction = tokenizer.decode(
            tokenizer.post_quant(selected.to(projected.dtype))
        )
    reconstruction = finish_uvg_reconstruction(reconstruction, prepared)
    if not math.isfinite(bits) or not torch.isfinite(reconstruction).all():
        raise RuntimeError("Non-finite entropy rate or reconstruction")
    save_reconstruction(reconstruction, recon_root)
    reference = prepared.reference
    rate_pixels = reference.shape[0] * reference.shape[-2] * reference.shape[-1]
    metric = lpips.LPIPS(net="vgg", version="0.1", verbose=False).to(device).eval()
    score = sequence_lpips(reference, reconstruction, metric, device)
    return {"bpp": bits / rate_pixels, "lpips": score}
