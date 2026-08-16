import pytest
import torch

from datasets.transforms import shared_crop
from tokenizer.cosmos_dv.preprocessing import pad_for_cosmos, unpad_reconstruction


def test_shared_crop_uses_same_coordinates_for_all_frames() -> None:
    base = torch.arange(8 * 10).reshape(1, 1, 8, 10)
    video = torch.cat([base, base + 1000], dim=0)
    cropped = shared_crop(video, 4, random_crop=True, generator=torch.Generator().manual_seed(7))
    assert cropped.shape == (2, 1, 4, 4)
    assert torch.equal(cropped[1] - cropped[0], torch.full((1, 4, 4), 1000))


def test_causal_padding_32_frames_and_1080p() -> None:
    video = torch.zeros(1, 3, 32, 1080, 1920)
    padded, info = pad_for_cosmos(video)
    assert padded.shape == (1, 3, 33, 1088, 1920)
    assert info.temporal_start == 0
    assert info.height_start == 4
    assert torch.all(padded[:, :, :, :4] == -1)
    restored = unpad_reconstruction(padded, info)
    assert restored.shape == video.shape


@pytest.mark.parametrize("frames,latent_frames", [(25, 7), (21, 6)])
def test_ar_protocol_lengths_are_not_temporally_padded(
    frames: int, latent_frames: int
) -> None:
    video = torch.zeros(1, 3, frames, 16, 16)
    padded, info = pad_for_cosmos(video, temporal_alignment=4)

    assert padded.shape == video.shape
    assert info.padded_frames == frames
    assert 1 + (info.padded_frames - 1) // 4 == latent_frames
    assert torch.equal(unpad_reconstruction(padded, info), video)
