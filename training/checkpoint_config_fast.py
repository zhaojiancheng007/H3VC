from __future__ import annotations

from copy import deepcopy
from typing import Any


RUNTIME_ONLY_CONFIG_PATHS = (("train", "num_workers"),)


def assert_fast_loader_resume_config(
    checkpoint_config: dict[str, Any],
    current_config: dict[str, Any],
    *,
    stage: str,
) -> None:
    checkpoint_scientific = deepcopy(checkpoint_config)
    current_scientific = deepcopy(current_config)
    for section, field in RUNTIME_ONLY_CONFIG_PATHS:
        del checkpoint_scientific[section][field]
        del current_scientific[section][field]
    if checkpoint_scientific != current_scientific:
        raise ValueError(f"{stage} resume scientific config changed")
