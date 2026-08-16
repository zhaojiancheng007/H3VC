from pathlib import Path

import torch

from entropy_model import CosmosFSQEntropyModel, EntropyModelConfig
from entropy_model.checkpoint import load_entropy_checkpoint
from training.checkpoint_ar_ddp import (
    DDPTrainingState,
    load_ddp_training_checkpoint,
    save_ddp_training_checkpoint,
)


def test_ddp_checkpoint_is_resumable_and_evaluator_compatible(tmp_path: Path) -> None:
    model = CosmosFSQEntropyModel(
        EntropyModelConfig(
            window_size=4,
            d_model=32,
            num_layers=1,
            num_heads=4,
            ffn_dim=64,
            max_latent_frames=2,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    loss = model(torch.randint(0, 64000, (1, 2, 8, 8))).bits_per_site
    loss.backward()
    optimizer.step()
    scheduler.step()

    path = tmp_path / "last.pt"
    state = DDPTrainingState(
        step=1,
        epoch=0,
        batch_in_epoch=2,
        max_steps=10,
        world_size=2,
        global_batch_size=8,
    )
    save_ddp_training_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        state=state,
        rng_states=[{"rank": 0}, {"rank": 1}],
    )

    payload = load_ddp_training_checkpoint(path, device="cpu")
    assert payload["training_state"] == {
        "step": 1,
        "epoch": 0,
        "batch_in_epoch": 2,
        "max_steps": 10,
        "world_size": 2,
        "global_batch_size": 8,
    }
    assert payload["scheduler"]["last_epoch"] == 1
    assert not (tmp_path / ".last.pt.tmp").exists()

    restored, evaluator_payload = load_entropy_checkpoint(path)
    assert evaluator_payload["step"] == 1
    for name, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)
