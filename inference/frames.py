"""Read one consecutive 96-frame RGB PNG sequence."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch


def load_png_sequence(frames_dir):
    frames = []
    for index in range(1, 97):
        path = Path(frames_dir) / f"{index:06d}.png"
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise ValueError(f"Expected RGB PNG: {path}")
            frames.append(torch.from_numpy(np.array(image)).permute(2, 0, 1))
    return torch.stack(frames)
