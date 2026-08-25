import torch

from entropy_model.config_categorical import CategoricalEntropyModelConfig
from entropy_model.model_categorical_dynamic_rank_reliable import (
    CategoricalEntropyModelDynamicRankReliable,
)
from entropy_model.model_categorical_temporal_window import (
    CategoricalEntropyModelTemporalWindow,
    sinusoidal_window_positions,
)


def build_models() -> tuple[
    CategoricalEntropyModelDynamicRankReliable,
    CategoricalEntropyModelTemporalWindow,
]:
    config = CategoricalEntropyModelConfig(
        codebook_size=4096,
        codebook_dim=16,
        window_size=2,
        d_model=8,
        num_layers=1,
        num_heads=2,
        ffn_dim=16,
        max_latent_frames=3,
    )
    parent = CategoricalEntropyModelDynamicRankReliable(config).eval()
    temporal = CategoricalEntropyModelTemporalWindow(config).eval()
    temporal.load_parent_state_dict(parent.state_dict())
    return parent, temporal


def test_sinusoidal_window_positions_support_arbitrary_grid() -> None:
    positions = sinusoidal_window_positions(
        3, 5, 8, device=torch.device("cpu"), dtype=torch.float32
    )
    assert positions.shape == (15, 8)
    assert not torch.equal(positions[0], positions[1])
    assert not torch.equal(positions[0], positions[5])


def test_zero_initialized_temporal_adapter_exactly_matches_parent() -> None:
    torch.manual_seed(7)
    parent, temporal = build_models()
    quantized = torch.randn(1, 16, 3, 4, 4)
    indices = torch.randint(4096, (1, 3, 4, 4))

    parent_output = parent(quantized, indices)
    temporal_output = temporal(quantized, indices)

    torch.testing.assert_close(temporal_output.logits, parent_output.logits, rtol=0, atol=0)
    torch.testing.assert_close(
        temporal_output.estimated_bits, parent_output.estimated_bits, rtol=0, atol=0
    )


def test_temporal_context_is_strictly_past_frame_and_cross_window() -> None:
    torch.manual_seed(11)
    _, model = build_models()
    torch.nn.init.normal_(model.temporal.cross_attention.output.weight)
    torch.nn.init.normal_(model.local.output_head.weight)
    quantized = torch.randn(1, 16, 3, 4, 4)
    reference, _ = model.predict(quantized)

    future_changed = quantized.clone()
    future_changed[:, :, 2] += 10.0
    future_logits, _ = model.predict(future_changed)
    torch.testing.assert_close(reference[:, :8], future_logits[:, :8])

    same_frame_other_window = quantized.clone()
    same_frame_other_window[:, :, 1, :2, 2:] += 10.0
    same_frame_logits, _ = model.predict(same_frame_other_window)
    torch.testing.assert_close(reference[0, 4:8], same_frame_logits[0, 4:8])

    past_other_window = quantized.clone()
    past_other_window[:, :, 0, :2, 2:] += 10.0
    past_logits, _ = model.predict(past_other_window)
    assert not torch.allclose(reference[0, 4:8], past_logits[0, 4:8])


def test_teacher_forward_matches_incremental_temporal_cache() -> None:
    torch.manual_seed(19)
    _, model = build_models()
    torch.nn.init.normal_(model.temporal.cross_attention.output.weight)
    torch.nn.init.normal_(model.local.output_head.weight)
    quantized = torch.randn(1, 16, 3, 4, 4)
    teacher_logits, blocks = model.predict(quantized)
    target_positions, source_positions = model.local._position_embeddings_for_blocks(
        blocks, model.local.code_projection.weight.dtype
    )

    temporal_memory = model.empty_temporal_memory(
        blocks.values.shape[0], dtype=blocks.values.dtype
    )
    local_cache = None
    previous_vectors = None
    incremental = []
    for frame_index in range(blocks.latent_frames):
        frame_vectors = []
        for rank in range(model.layout.num_groups):
            block_index = frame_index * model.layout.num_groups + rank
            if block_index == 0:
                inputs = model.local.bos.expand(blocks.values.shape[0], -1, -1)
            else:
                inputs = model.local.code_projection(previous_vectors)
                status = torch.zeros(
                    previous_vectors.shape[:2], dtype=torch.long
                )
                inputs = inputs + model.local.source_status_embedding(status)
            inputs = inputs + target_positions[:, block_index] + source_positions[:, block_index]
            logits, local_cache = model.forward_block(
                inputs, local_cache, temporal_memory
            )
            incremental.append(logits)
            previous_vectors = blocks.values[:, block_index]
            frame_vectors.append(previous_vectors)
        temporal_memory = model.append_completed_frame(
            temporal_memory,
            torch.stack(frame_vectors, dim=1),
            blocks,
            frame_index,
        )

    incremental_logits = torch.stack(incremental, dim=1)
    torch.testing.assert_close(incremental_logits, teacher_logits, rtol=1e-5, atol=1e-5)
