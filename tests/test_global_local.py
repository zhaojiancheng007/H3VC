import torch

from entropy_model.config_categorical import CategoricalEntropyModelConfig
from entropy_model.config_local_global_memory import LocalGlobalMemoryEntropyConfig
from entropy_model.model_categorical import CategoricalEntropyModel
from entropy_model.model_local_global_memory import LocalGlobalMemoryEntropyModel


def tiny_model() -> LocalGlobalMemoryEntropyModel:
    torch.manual_seed(7)
    model = LocalGlobalMemoryEntropyModel(
        LocalGlobalMemoryEntropyConfig(
            codebook_size=4096, codebook_dim=16, window_size=8,
            d_model=32, num_layers=2, num_heads=4, ffn_dim=64,
            max_latent_frames=4, memory_tokens=512, summary_tokens=32,
            global_tokens=32, global_read_layers=2,
        )
    )
    torch.nn.init.normal_(model.output_head.weight, std=0.02)
    return model.eval()


def sample(frames: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    quantized = torch.randn(1, 16, frames, 16, 16)
    indices = torch.randint(0, 4096, (1, frames, 16, 16))
    return quantized, indices








def test_future_frames_cannot_change_past_logits() -> None:
    model = tiny_model()
    quantized, _ = sample()
    changed = quantized.clone()
    changed[:, :, 2] += 3.0
    with torch.no_grad():
        before, _ = model.predict(quantized, k=1)
        after, _ = model.predict(changed, k=1)
    torch.testing.assert_close(before[:, :8], after[:, :8], atol=0.0, rtol=0.0)


def test_current_group_only_changes_later_rank_logits() -> None:
    model = tiny_model()
    quantized, _ = sample(frames=1)
    changed = quantized.clone()
    changed[:, :, 0, 1::2, 0::2] += 3.0
    with torch.no_grad():
        before, _ = model.predict(quantized, k=1)
        after, _ = model.predict(changed, k=1)
    torch.testing.assert_close(before[:, :3], after[:, :3], atol=0.0, rtol=0.0)
    assert not torch.equal(before[:, 3], after[:, 3])


def test_same_window_uses_all_previous_frames_only() -> None:
    model = tiny_model()
    model.read_all_global = lambda banks: torch.zeros(
        banks.shape[0], banks.shape[1], model.config.global_tokens, model.config.d_model
    )
    quantized, _ = sample(frames=3)
    changed = quantized.clone()
    changed[:, :, 0, 0, 0] += 5.0
    with torch.no_grad():
        before, _ = model.predict(quantized, k=1)
        after, _ = model.predict(changed, k=1)
    difference = (before[:, 8:12] - after[:, 8:12]).abs().sum(dim=(1, 2, 3))
    assert difference[0] > 0
    torch.testing.assert_close(difference[1:], torch.zeros_like(difference[1:]))


def test_local_forward_is_exactly_the_canonical_model() -> None:
    model = tiny_model()
    config = CategoricalEntropyModelConfig(
        codebook_size=model.config.codebook_size,
        codebook_dim=model.config.codebook_dim,
        window_size=model.config.window_size,
        d_model=model.config.d_model,
        num_layers=model.config.num_layers,
        num_heads=model.config.num_heads,
        ffn_dim=model.config.ffn_dim,
        max_latent_frames=model.config.max_latent_frames,
    )
    canonical = CategoricalEntropyModel(config).eval()
    canonical.load_state_dict(
        {name: model.state_dict()[name] for name in canonical.state_dict()}
    )
    torch.nn.init.zeros_(model.global_fusion.cross_attn.out_proj.weight)
    torch.nn.init.zeros_(model.global_fusion.cross_attn.out_proj.bias)
    torch.nn.init.zeros_(model.global_fusion.ffn[-1].weight)
    quantized, _ = sample()
    with torch.no_grad():
        expected, _ = canonical.predict(quantized, k=1)
        actual, _ = model.predict(quantized, k=1)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
