import torch
from torch import nn

from training.gan_stage3 import (
    PatchDiscriminator,
    discriminator_microbatch_backward,
    generator_adversarial_loss,
    hinge_discriminator_loss,
)


def test_generator_backward_preserves_batchnorm_and_skips_discriminator_grad() -> None:
    torch.manual_seed(11)
    discriminator = PatchDiscriminator(base_channels=8, num_layers=2).train()
    fake_frames = torch.randn(2, 3, 64, 64, requires_grad=True)
    batchnorm_state = {
        name: buffer.clone()
        for name, buffer in discriminator.named_buffers()
    }

    loss = generator_adversarial_loss(discriminator, fake_frames)
    loss.backward()

    assert discriminator.training
    assert fake_frames.grad is not None
    assert torch.count_nonzero(fake_frames.grad)
    assert all(parameter.requires_grad for parameter in discriminator.parameters())
    assert all(parameter.grad is None for parameter in discriminator.parameters())
    for name, buffer in discriminator.named_buffers():
        assert torch.equal(buffer, batchnorm_state[name])


def test_discriminator_backward_accumulates_every_microbatch() -> None:
    torch.manual_seed(13)
    discriminator = nn.Conv2d(3, 1, kernel_size=1, bias=False)
    reference = nn.Conv2d(3, 1, kernel_size=1, bias=False)
    reference.load_state_dict(discriminator.state_dict())
    microbatches = [
        (torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4, requires_grad=True))
        for _ in range(2)
    ]

    losses = [
        discriminator_microbatch_backward(
            discriminator,
            real,
            fake,
            loss_weight=0.1,
            gradient_accumulation=2,
        )
        for real, fake in microbatches
    ]

    reference_losses = []
    for real, fake in microbatches:
        logits = reference(torch.cat([real, fake.detach()], dim=0))
        reference_losses.append(
            0.1 * hinge_discriminator_loss(logits[:1], logits[1:])
        )
    expected_gradient = torch.autograd.grad(
        sum(reference_losses) / 2, reference.weight
    )[0]

    assert torch.allclose(discriminator.weight.grad, expected_gradient)
    assert torch.allclose(torch.stack(losses).mean(), sum(reference_losses) / 2)
    assert all(fake.grad is None for _, fake in microbatches)
