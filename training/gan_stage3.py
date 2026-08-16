from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PatchDiscriminator(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        kernel_size = 4
        padding = 1
        normalization = nn.BatchNorm2d
        use_bias = normalization != nn.BatchNorm2d
        layers: list[nn.Module] = [
            nn.Conv2d(
                input_channels,
                base_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        channel_multiplier = 1
        for layer_index in range(1, num_layers):
            previous_multiplier = channel_multiplier
            channel_multiplier = min(2**layer_index, 8)
            layers.extend(
                [
                    nn.Conv2d(
                        base_channels * previous_multiplier,
                        base_channels * channel_multiplier,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=padding,
                        bias=use_bias,
                    ),
                    normalization(base_channels * channel_multiplier),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        previous_multiplier = channel_multiplier
        channel_multiplier = min(2**num_layers, 8)
        layers.extend(
            [
                nn.Conv2d(
                    base_channels * previous_multiplier,
                    base_channels * channel_multiplier,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    bias=use_bias,
                ),
                normalization(base_channels * channel_multiplier),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(
                    base_channels * channel_multiplier,
                    1,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                ),
            ]
        )
        self.main = nn.Sequential(*layers)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.zeros_(module.bias)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.main(frames)


def select_gan_frames(
    video: torch.Tensor,
    reconstruction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if video.shape != reconstruction.shape or video.ndim != 5:
        raise ValueError(
            "GAN inputs must have matching [B,C,T,H,W] shapes, got "
            f"{tuple(video.shape)} and {tuple(reconstruction.shape)}"
        )
    batch_size, channels, frames, height, width = video.shape
    frame_indices = torch.randint(frames, (batch_size,), device=video.device)
    batch_indices = torch.arange(batch_size, device=video.device)
    real = video.permute(0, 2, 1, 3, 4)[batch_indices, frame_indices]
    fake = reconstruction.permute(0, 2, 1, 3, 4)[batch_indices, frame_indices]
    expected_shape = (batch_size, channels, height, width)
    if real.shape != expected_shape or fake.shape != expected_shape:
        raise RuntimeError("Selected GAN frame shape changed")
    return real, fake


def hinge_generator_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    return -logits_fake.float().mean()


def hinge_discriminator_loss(
    logits_real: torch.Tensor,
    logits_fake: torch.Tensor,
) -> torch.Tensor:
    real_loss = F.relu(1.0 - logits_real.float()).mean()
    fake_loss = F.relu(1.0 + logits_fake.float()).mean()
    return 0.5 * (real_loss + fake_loss)


def generator_adversarial_loss(
    discriminator: nn.Module,
    fake_frames: torch.Tensor,
) -> torch.Tensor:
    was_training = discriminator.training
    discriminator.eval()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    loss = hinge_generator_loss(discriminator(fake_frames))
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)
    discriminator.train(was_training)
    return loss


def discriminator_microbatch_backward(
    discriminator: nn.Module,
    real_frames: torch.Tensor,
    fake_frames: torch.Tensor,
    *,
    loss_weight: float,
    gradient_accumulation: int,
) -> torch.Tensor:
    logits = discriminator(
        torch.cat([real_frames, fake_frames.detach()], dim=0)
    )
    split = real_frames.shape[0]
    weighted_loss = loss_weight * hinge_discriminator_loss(
        logits[:split], logits[split:]
    )
    (weighted_loss / gradient_accumulation).backward()
    return weighted_loss.detach()
