from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import torch

from tokenizer.wan_vq.parallel_wan import parallel_decode, parallel_encode


WAN_REPO = Path(__file__).resolve().parents[1] / "Wan2.1"
sys.path.insert(0, str(WAN_REPO))

from wan.modules.vae import WanVAE_  # noqa: E402


def _tiny_wan() -> WanVAE_:
    return WanVAE_(
        dim=4,
        z_dim=2,
        dim_mult=[1, 2, 2, 2],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )


def test_full_sequence_matches_streaming_outputs_and_gradients() -> None:
    torch.manual_seed(11)
    streaming = _tiny_wan().double()
    parallel = deepcopy(streaming)
    scale = [torch.zeros(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64)]
    streaming_input = torch.randn(
        1, 3, 9, 16, 16, dtype=torch.float64, requires_grad=True
    )
    parallel_input = streaming_input.detach().clone().requires_grad_(True)

    streaming_latent = streaming.encode(streaming_input, scale)
    streaming_reconstruction = streaming.decode(streaming_latent, scale)
    parallel_latent = parallel_encode(parallel, parallel_input, scale)
    parallel_reconstruction = parallel_decode(parallel, parallel_latent, scale)

    torch.testing.assert_close(parallel_latent, streaming_latent, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(
        parallel_reconstruction, streaming_reconstruction, rtol=1e-9, atol=1e-9
    )

    streaming_reconstruction.square().mean().backward()
    parallel_reconstruction.square().mean().backward()
    torch.testing.assert_close(
        parallel_input.grad, streaming_input.grad, rtol=1e-8, atol=1e-10
    )
    streaming_parameters = dict(streaming.named_parameters())
    parallel_parameters = dict(parallel.named_parameters())
    assert streaming_parameters.keys() == parallel_parameters.keys()
    for name in streaming_parameters:
        torch.testing.assert_close(
            parallel_parameters[name].grad,
            streaming_parameters[name].grad,
            rtol=1e-8,
            atol=1e-10,
            msg=lambda message, parameter_name=name: f"{parameter_name}: {message}",
        )
