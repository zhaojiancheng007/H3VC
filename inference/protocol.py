"""The original 96-source-frame / 107-model-frame inference protocol."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class EvaluationInput:
    reference: torch.Tensor
    model_input: torch.Tensor


def prepare_uvg_input(video):
    if video.shape[0] != 96:
        raise ValueError("Expected 96 input frames")
    reference = video
    height, width = video.shape[-2:]
    video = F.pad(video, (0, (-width) % 16, 0, (-height) % 16), mode="replicate")
    video = torch.cat((video, video[-1:].expand(11, -1, -1, -1)), dim=0)
    model_input = video.permute(1, 0, 2, 3).unsqueeze(0).mul(2.0).sub(1.0)
    return EvaluationInput(reference, model_input)


def finish_uvg_reconstruction(reconstruction, prepared):
    height, width = prepared.reference.shape[-2:]
    reconstruction = reconstruction[:, :, :96, :height, :width]
    return (reconstruction.float().clamp(-1.0, 1.0).add(1.0).div(2.0)
            .squeeze(0).permute(1, 0, 2, 3).cpu())
