import torch

from tokenizer.stage_backbones import checkpoint_input_frames_match
from training.uvg_protocol import prepare_uvg_input, uvg_metric_frames


def test_h3_uvg_pads_96_frames_to_aligned_spatial_input() -> None:
    video = torch.arange(96, dtype=torch.float32).view(96, 1, 1, 1).expand(-1, 3, 33, 49)

    prepared = prepare_uvg_input(video, "minimax_h3")

    assert uvg_metric_frames("minimax_h3") == 96
    assert prepared.reference.shape == (96, 3, 33, 49)
    assert prepared.model_input.shape == (1, 3, 96, 48, 64)
    torch.testing.assert_close(prepared.reference, video)
    torch.testing.assert_close(
        prepared.model_input[:, :, :, :33, :49],
        video.permute(1, 0, 2, 3).unsqueeze(0).mul(2.0).sub(1.0),
    )
    torch.testing.assert_close(
        prepared.model_input[:, :, :, 33:, :],
        prepared.model_input[:, :, :, 32:33, :].expand(-1, -1, -1, 15, -1),
    )
    torch.testing.assert_close(
        prepared.model_input[:, :, :, :, 49:],
        prepared.model_input[:, :, :, :, 48:49].expand(-1, -1, -1, -1, 15),
    )


def test_non_h3_uvg_keeps_97_forward_and_96_metrics() -> None:
    video = torch.zeros(97, 3, 32, 48)

    prepared = prepare_uvg_input(video, "wanvae")

    assert uvg_metric_frames("wanvae") == 96
    assert prepared.reference.shape[0] == 96
    assert prepared.model_input.shape[2] == 97


def test_only_old_h3_90_frame_checkpoints_migrate_to_96() -> None:
    assert checkpoint_input_frames_match("minimax_h3", 96)
    assert checkpoint_input_frames_match("minimax_h3", 90)
    assert not checkpoint_input_frames_match("minimax_h3", 107)
    assert not checkpoint_input_frames_match("wanvae", 107)
