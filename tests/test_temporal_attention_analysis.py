import torch

from scripts.analyze_stage2_temporal_attention import (
    minimum_recent_frame_radius,
    oracle_retained_density,
    shifted_source_block,
)
from training.temporal_attention_mask_ablation import temporal_radius_attention_mask


def test_oracle_retained_density_uses_minimum_top_probability_prefix() -> None:
    probabilities = torch.tensor([[0.50, 0.25, 0.15, 0.10]])
    density = oracle_retained_density(probabilities, target_mass=0.9)
    assert torch.equal(density, torch.tensor([0.75]))


def test_recent_frame_radius_accumulates_from_latest_frame() -> None:
    chronological_frame_mass = torch.tensor([[0.10, 0.20, 0.70]])
    radius = minimum_recent_frame_radius(chronological_frame_mass, target_mass=0.9)
    assert torch.equal(radius, torch.tensor([2]))


def test_shifted_key_block_maps_to_previous_source_block() -> None:
    assert shifted_source_block(1) == (0, 0)
    assert shifted_source_block(4) == (0, 3)
    assert shifted_source_block(5) == (1, 0)
def test_temporal_radius_mask_keeps_same_frame_groups_and_recent_history() -> None:
    mask = temporal_radius_attention_mask(
        num_frames=3,
        num_groups=4,
        sites_per_block=1,
        radius=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )[0, 0]
    allowed = mask == 0

    target_t2_g0 = 8
    assert allowed[target_t2_g0, 0]
    assert not allowed[target_t2_g0, 4]
    assert allowed[target_t2_g0, 5:9].all()

    target_t2_g2 = 10
    assert allowed[target_t2_g2, 9:11].all()
    assert not allowed[target_t2_g2, 11]
