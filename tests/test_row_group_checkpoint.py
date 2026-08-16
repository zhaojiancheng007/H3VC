from pathlib import Path

import torch

from entropy_model.checkpoint_row_group import load_row_group_entropy_checkpoint
from entropy_model.config_row_group import RowGroupEntropyModelConfig
from entropy_model.model_quant_feature_row_group import CosmosFSQRowGroupEntropyModel
from training.checkpoint_ar_ddp import (
    DDPTrainingState,
    save_ddp_training_checkpoint,
)


def test_row_group_training_checkpoint_loads_for_eval(tmp_path: Path) -> None:
    model = CosmosFSQRowGroupEntropyModel(
        RowGroupEntropyModelConfig(
            d_model=16,
            num_layers=1,
            num_heads=2,
            ffn_dim=32,
            max_latent_frames=1,
            max_latent_height=4,
            max_latent_width=4,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "row_group.pt"
    save_ddp_training_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        state=DDPTrainingState(
            step=0,
            epoch=0,
            batch_in_epoch=0,
            max_steps=10,
            world_size=1,
            global_batch_size=1,
        ),
        rng_states=[{}],
    )

    restored, payload = load_row_group_entropy_checkpoint(path)

    assert payload["step"] == 0
    for name, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)
