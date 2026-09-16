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


def test_shapes_and_bits() -> None:
    model = tiny_model()
    quantized, indices = sample()
    output = model(quantized, indices, k=1)
    assert output.logits.shape == (4, 12, 16, 4096)
    assert output.estimated_bits.shape == (1,)
    assert output.bits_per_index.ndim == 0
    assert torch.isfinite(output.bits_per_index)


def test_rectangular_benchmark_latent_grid() -> None:
    model = tiny_model()
    quantized = torch.randn(1, 16, 2, 24, 32)
    indices = torch.randint(0, 4096, (1, 2, 24, 32))
    with torch.no_grad():
        output = model(quantized, indices, k=1)
        state = model.init_state(1, device=torch.device("cpu"), dtype=output.logits.dtype)
        online = model.predict_rank(state, quantized[:, :, 0], rank=0)
        state = model.update_state(state, quantized[:, :, 0])
    assert output.logits.shape == (12, 8, 16, 4096)
    assert online.shape == (1, 12, 16, 4096)
    assert state.local_history is not None


def test_vectorized_banks_equal_online_fifo_state() -> None:
    model = tiny_model()
    quantized, _ = sample()
    vectors = model._frame_vectors(quantized)
    summaries = model.summarize_frames(vectors)
    banks = model.memory_banks(summaries)
    state = model.init_state(1, device=quantized.device, dtype=vectors.dtype)
    for frame in range(quantized.shape[2]):
        torch.testing.assert_close(banks[:, frame], state.memory)
        state = model.update_state(state, quantized[:, :, frame])


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


def test_global_reader_has_exactly_two_layers_and_no_bitstream_tokens() -> None:
    model = tiny_model()
    assert len(model.global_reader) == 2
    state = model.init_state(2, device=torch.device("cpu"), dtype=torch.float32)
    global_tokens = model.read_global(state.memory)
    assert state.memory.shape == (2, 512, 32)
    assert global_tokens.shape == (2, 32, 32)


def test_vectorized_logits_equal_online_rank_predictions() -> None:
    model = tiny_model()
    quantized, _ = sample(frames=2)
    with torch.no_grad():
        expected, blocks = model.predict(quantized, k=1)
        assert blocks.num_windows == 4
        assert expected.shape == (4, 8, 16, 4096)
        state = model.init_state(1, device=quantized.device, dtype=expected.dtype)
        for frame in range(quantized.shape[2]):
            decoded = quantized[:, :, frame]
            for rank in range(4):
                actual = model.predict_rank(state, decoded, rank=rank)
                torch.testing.assert_close(
                    actual, expected[:, frame * 4 + rank].reshape(1, 4, 16, 4096),
                    atol=1e-5, rtol=1e-5
                )
            state = model.update_state(state, decoded)


def test_rate_loss_trains_entropy_summary_and_global_reader() -> None:
    model = tiny_model().train()
    quantized, indices = sample(frames=2)
    model(quantized, indices).bits_per_index.backward()
    assert model.transformer.layers[0].attention.qkv.weight.grad is not None
    assert model.summary_block.cross_attn.in_proj_weight.grad is not None
    assert model.global_reader[0].cross_attn.in_proj_weight.grad is not None
    assert model.global_fusion.cross_attn.in_proj_weight.grad is not None
