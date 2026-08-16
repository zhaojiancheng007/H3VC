from __future__ import annotations

from pathlib import Path

import torch

from .checkpoint import CHECKPOINT_VERSION
from .config_row_group import RowGroupEntropyModelConfig
from .model_quant_feature_row_group import CosmosFSQRowGroupEntropyModel


def load_row_group_entropy_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[CosmosFSQRowGroupEntropyModel, dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {payload.get('version')}, "
            f"expected {CHECKPOINT_VERSION}"
        )
    config_values = dict(payload["model_config"])
    config_values["levels"] = tuple(config_values["levels"])
    model = CosmosFSQRowGroupEntropyModel(
        RowGroupEntropyModelConfig(**config_values)
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model, payload
