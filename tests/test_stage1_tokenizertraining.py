import torch
import torch.nn as nn

from scripts.stage1_tokenizertraining import (
    CHECKPOINT_INTERVAL,
    LPIPS_WEIGHT,
    MSE_WEIGHT,
    save_checkpoint,
)
from tokenizer.stage1_vq import (
    QUANTIZER_GEOMETRY,
    EMAVectorQuantizer,
    Stage1VQTokenizer,
)


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv3d(3, 16, kernel_size=1)
        self.decoder = nn.Conv3d(16, 3, kernel_size=1)

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        return self.encoder(video)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)


def test_stage1_ema_vq_forward_and_gradient_path() -> None:
    torch.manual_seed(3)
    model = Stage1VQTokenizer(TinyBackbone(), codebook_size=4, codebook_dim=16)
    video = torch.randn(1, 3, 1, 2, 2)

    output = model(video)
    loss = output.reconstruction.square().mean() + output.commitment_loss
    loss.backward()

    assert output.indices.shape == (1, 1, 2, 2)
    assert model.quantizer.initted.item()
    assert model.backbone.encoder.weight.grad is not None
    assert model.backbone.decoder.weight.grad is not None
    assert model.pre_quant.weight.grad is not None
    assert model.post_quant.weight.grad is not None
    assert list(model.quantizer.parameters()) == []


def test_identity_initialized_projections_preserve_raw_latent() -> None:
    model = Stage1VQTokenizer(TinyBackbone(), codebook_size=4, codebook_dim=16)
    latent = torch.randn(1, 16, 2, 3, 4)

    projected = model.pre_quant(latent)
    reconstructed_latent = model.post_quant(projected)

    assert QUANTIZER_GEOMETRY == "raw_l2"
    torch.testing.assert_close(projected, latent, rtol=0, atol=0)
    torch.testing.assert_close(reconstructed_latent, latent, rtol=0, atol=0)


def test_eval_does_not_update_ema_codebook() -> None:
    quantizer = EMAVectorQuantizer(codebook_size=4, codebook_dim=16)
    projected = torch.randn(1, 16, 1, 2, 2)
    quantizer(projected)
    before = {name: value.clone() for name, value in quantizer.state_dict().items()}

    quantizer.eval()
    quantizer(projected * 2.0)

    for name, value in quantizer.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_checkpoint_contains_general_stage1_state(tmp_path) -> None:
    model = Stage1VQTokenizer(TinyBackbone(), codebook_size=4, codebook_dim=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    path = tmp_path / "step_0005000.pt"

    save_checkpoint(
        path,
        model,
        optimizer,
        "wanvae",
        input_frames=97,
        latent_channels=16,
        codebook_dim=16,
        config={"train": {}},
        step=5000,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert CHECKPOINT_INTERVAL == 5000
    assert MSE_WEIGHT == LPIPS_WEIGHT == 1.0
    assert payload["tokenizer"] == "wanvae"
    assert payload["step"] == 5000
    assert payload["quantizer_geometry"] == "raw_l2"
    assert set(payload["model"]) == set(model.state_dict())
