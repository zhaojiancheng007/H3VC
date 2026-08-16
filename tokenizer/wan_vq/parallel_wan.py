from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def _run_module(
    module: nn.Module, x: torch.Tensor, *, activation_checkpointing: bool
) -> torch.Tensor:
    if activation_checkpointing:
        return checkpoint(module, x, use_reentrant=False)
    return module(x)


def _spatial_resample(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    batch, channels, frames, height, width = x.shape
    images = x.permute(0, 2, 1, 3, 4).reshape(
        batch * frames, channels, height, width
    )
    images = module.resample(images)
    output_channels, output_height, output_width = images.shape[1:]
    return images.reshape(
        batch, frames, output_channels, output_height, output_width
    ).permute(0, 2, 1, 3, 4).contiguous()


def _parallel_resample(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if module.mode == "downsample3d":
        x = _spatial_resample(module, x)
        first_frame = x[:, :, :1]
        if x.shape[2] == 1:
            return first_frame
        return torch.cat((first_frame, module.time_conv(x)), dim=2)

    if module.mode == "upsample3d":
        first_frame = x[:, :, :1]
        remaining_frames = x[:, :, 1:]
        if remaining_frames.shape[2] > 0:
            batch, channels, frames, height, width = remaining_frames.shape
            remaining_frames = module.time_conv(remaining_frames)
            remaining_frames = remaining_frames.reshape(
                batch, 2, channels, frames, height, width
            )
            remaining_frames = torch.stack(
                (remaining_frames[:, 0], remaining_frames[:, 1]), dim=3
            ).reshape(batch, channels, 2 * frames, height, width)
            x = torch.cat((first_frame, remaining_frames), dim=2)
        return _spatial_resample(module, x)

    return _spatial_resample(module, x)


def _run_blocks(
    blocks: nn.Module, x: torch.Tensor, *, activation_checkpointing: bool
) -> torch.Tensor:
    for module in blocks:
        if getattr(module, "mode", None) is not None:
            if activation_checkpointing:
                x = checkpoint(
                    lambda value, current=module: _parallel_resample(current, value),
                    x,
                    use_reentrant=False,
                )
            else:
                x = _parallel_resample(module, x)
        else:
            x = _run_module(
                module, x, activation_checkpointing=activation_checkpointing
            )
    return x


def parallel_encode(
    wan: nn.Module,
    video: torch.Tensor,
    scale: list[torch.Tensor],
    *,
    activation_checkpointing: bool = False,
) -> torch.Tensor:
    x = _run_module(
        wan.encoder.conv1, video, activation_checkpointing=activation_checkpointing
    )
    x = _run_blocks(
        wan.encoder.downsamples, x,
        activation_checkpointing=activation_checkpointing,
    )
    x = _run_module(
        wan.encoder.middle, x, activation_checkpointing=activation_checkpointing
    )
    x = _run_module(
        wan.encoder.head, x, activation_checkpointing=activation_checkpointing
    )
    moments = _run_module(
        wan.conv1, x, activation_checkpointing=activation_checkpointing
    )
    mean, _ = moments.chunk(2, dim=1)
    return (mean - scale[0].view(1, wan.z_dim, 1, 1, 1)) * scale[1].view(
        1, wan.z_dim, 1, 1, 1
    )


def parallel_decode(
    wan: nn.Module,
    latent: torch.Tensor,
    scale: list[torch.Tensor],
    *,
    activation_checkpointing: bool = False,
) -> torch.Tensor:
    x = latent / scale[1].view(1, wan.z_dim, 1, 1, 1)
    x = x + scale[0].view(1, wan.z_dim, 1, 1, 1)
    x = _run_module(wan.conv2, x, activation_checkpointing=activation_checkpointing)
    x = _run_module(
        wan.decoder.conv1, x, activation_checkpointing=activation_checkpointing
    )
    x = _run_module(
        wan.decoder.middle, x, activation_checkpointing=activation_checkpointing
    )
    x = _run_blocks(
        wan.decoder.upsamples, x,
        activation_checkpointing=activation_checkpointing,
    )
    return _run_module(
        wan.decoder.head, x, activation_checkpointing=activation_checkpointing
    )
