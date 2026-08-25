from __future__ import annotations

from copy import deepcopy
from typing import Any


RUNTIME_ONLY_CONFIG_PATHS = (("train", "num_workers"),)
LEGACY_STAGE3_CONFIG_PATHS = (
    ("optimizer", "discriminator_learning_rate"),
    ("loss", "gan_start_step"),
    ("loss", "gan_weight"),
)


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
    for section, field in LEGACY_STAGE3_CONFIG_PATHS:
        checkpoint_scientific.get(section, {}).pop(field, None)
        current_scientific.get(section, {}).pop(field, None)
    if checkpoint_scientific != current_scientific:
        raise ValueError(f"{stage} resume scientific config changed")
