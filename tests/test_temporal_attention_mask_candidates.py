import torch

from entropy_model.config_categorical import CategoricalEntropyModelConfig
from entropy_model.model_categorical_context_corruption import (
    CategoricalEntropyModelContextCorruption,
)
from training.temporal_attention_mask_candidates import (
    candidate_layer_specs,
    logits_with_temporal_mask_candidate,
    temporal_hybrid_attention_mask,
)


def test_hybrid_mask_keeps_recent_frames_and_five_frame_anchors() -> None:
    mask = temporal_hybrid_attention_mask(
        12,
        4,
        1,
        radius=4,
        anchor_stride=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )[0, 0]
    allowed = mask == 0
    target = 11 * 4

    assert allowed[target, 0]
    assert allowed[target, 7 * 4 + 1 : 11 * 4 + 1].all()
    assert allowed[target, 6 * 4 + 1 : 7 * 4 + 1].all()
    assert allowed[target, 1 * 4 + 1 : 2 * 4 + 1].all()
    assert not allowed[target, 0 * 4 + 1 : 1 * 4 + 1].any()
    assert not allowed[target, target + 1 :].any()


def test_layerwise_candidate_matches_eight_layer_design() -> None:
    specs = candidate_layer_specs("layerwise_anchor_5", 8)

    assert specs[0] == (None, None)
    assert specs[3] == (4, 5)
    assert specs[-1] == (16, 5)


def test_dense_candidate_matches_original_forward() -> None:
    torch.manual_seed(73)
    config = CategoricalEntropyModelConfig(
        codebook_dim=16,
        window_size=2,
        d_model=8,
        num_layers=1,
        num_heads=2,
        ffn_dim=16,
        max_latent_frames=3,
    )
    model = CategoricalEntropyModelContextCorruption(config).eval()
    selected = torch.randn(1, 16, 3, 2, 2)

    expected, _ = model.predict(selected)
    actual, allowed, dense_allowed = logits_with_temporal_mask_candidate(
        model, selected, "dense"
    )

    assert torch.equal(actual, expected)
    assert allowed == dense_allowed
