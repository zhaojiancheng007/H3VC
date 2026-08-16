import pytest
import torch

from entropy_model.layout import FourGroupBlockLayout


@pytest.mark.parametrize("frames,height,width", [(6, 8, 8), (7, 16, 24), (9, 32, 32)])
def test_four_group_layout_round_trip(frames: int, height: int, width: int) -> None:
    layout = FourGroupBlockLayout(window_size=8)
    grid = torch.arange(frames * height * width * 6).reshape(1, frames, height, width, 6)
    blocks = layout.to_blocks(grid)

    assert blocks.values.shape == (
        (height // 8) * (width // 8),
        frames * 4,
        16,
        6,
    )
    assert torch.equal(layout.from_blocks(blocks), grid)


def test_four_group_layout_rejects_partial_windows() -> None:
    layout = FourGroupBlockLayout(window_size=8)
    with pytest.raises(ValueError, match="must be divisible"):
        layout.to_blocks(torch.zeros(1, 2, 10, 8, 6))
