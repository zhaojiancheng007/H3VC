import copy
from pathlib import Path

import pytest
import torch

from scripts.stage2 import scheduler_scale as stage2_scheduler_scale
from scripts.stage3 import scheduler_scale as stage3_scheduler_scale
from training.config_stage1_cosine import (
    learning_rate_at_step as stage1_learning_rate,
    load_stage1_training_config,
)
from training.config_stage2_cosine import (
    learning_rate_at_step as stage2_learning_rate,
    load_stage2_config,
)
from training.config_stage3_cosine import (
    cosine_learning_rates_at_step,
    load_stage3_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_cosine_configs_have_exact_endpoints_and_decrease() -> None:
    stage1 = load_stage1_training_config(ROOT / "configs/stage1.yaml")
    stage2 = load_stage2_config(ROOT / "configs/stage2.yaml")
    stage3 = load_stage3_config(ROOT / "configs/stage3.yaml")

    stage1_rates = [
        stage1_learning_rate(stage1.optimizer, stage1.train.max_steps, step)
        for step in (1, 30_000, 60_000)
    ]
    stage2_rates = [
        stage2_learning_rate(stage2.optimizer, stage2.train.max_steps, step)
        for step in (1, 30_000, 60_000)
    ]
    stage3_rates = [
        cosine_learning_rates_at_step(stage3, step)
        for step in (1, 30_000, 60_000)
    ]

    assert stage1_rates[0] == stage1.optimizer.learning_rate
    assert stage1_rates[-1] == stage1.optimizer.eta_min
    assert stage2_rates[0] == stage2.optimizer.learning_rate
    assert stage2_rates[-1] == stage2.optimizer.eta_min
    assert stage3_rates[0] == (
        stage3.optimizer.tokenizer_learning_rate,
        stage3.optimizer.entropy_learning_rate,
    )
    assert stage3_rates[-1] == (
        stage3.optimizer.tokenizer_eta_min,
        stage3.optimizer.entropy_eta_min,
    )
    assert stage1_rates == sorted(stage1_rates, reverse=True)
    assert stage2_rates == sorted(stage2_rates, reverse=True)
    assert [rate[0] for rate in stage3_rates] == sorted(
        (rate[0] for rate in stage3_rates), reverse=True
    )
    assert [rate[1] for rate in stage3_rates] == sorted(
        (rate[1] for rate in stage3_rates), reverse=True
    )


def test_lambda_lr_scales_match_cosine_step_definition() -> None:
    stage2 = load_stage2_config(ROOT / "configs/stage2.yaml")
    stage3 = load_stage3_config(ROOT / "configs/stage3.yaml")

    assert stage2_scheduler_scale(stage2, 0) == pytest.approx(1.0)
    assert stage2_scheduler_scale(stage2, stage2.train.max_steps - 1) == pytest.approx(
        stage2.optimizer.eta_min / stage2.optimizer.learning_rate
    )
    assert stage3_scheduler_scale(stage3, 0, 0) == pytest.approx(1.0)
    assert stage3_scheduler_scale(stage3, 0, 1) == pytest.approx(1.0)
    assert stage3_scheduler_scale(
        stage3, stage3.train.max_steps - 1, 0
    ) == pytest.approx(
        stage3.optimizer.tokenizer_eta_min
        / stage3.optimizer.tokenizer_learning_rate
    )
    assert stage3_scheduler_scale(
        stage3, stage3.train.max_steps - 1, 1
    ) == pytest.approx(
        stage3.optimizer.entropy_eta_min
        / stage3.optimizer.entropy_learning_rate
    )


def test_stage2_scheduler_resume_preserves_next_step_learning_rate() -> None:
    config = load_stage2_config(ROOT / "configs/stage2.yaml")
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW(
        [parameter], lr=config.optimizer.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: stage2_scheduler_scale(config, step)
    )

    completed_steps = 37
    for _ in range(completed_steps):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()

    optimizer_state = copy.deepcopy(optimizer.state_dict())
    scheduler_state = copy.deepcopy(scheduler.state_dict())

    resumed_parameter = torch.nn.Parameter(torch.tensor(0.0))
    resumed_optimizer = torch.optim.AdamW(
        [resumed_parameter], lr=config.optimizer.learning_rate
    )
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(
        resumed_optimizer, lambda step: stage2_scheduler_scale(config, step)
    )
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler_state)

    expected_next_rate = stage2_learning_rate(
        config.optimizer, config.train.max_steps, completed_steps + 1
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected_next_rate)
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(expected_next_rate)

    parameter.grad = torch.ones_like(parameter)
    resumed_parameter.grad = torch.ones_like(resumed_parameter)
    optimizer.step()
    scheduler.step()
    resumed_optimizer.step()
    resumed_scheduler.step()

    expected_following_rate = stage2_learning_rate(
        config.optimizer, config.train.max_steps, completed_steps + 2
    )
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(
        optimizer.param_groups[0]["lr"]
    )
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(
        expected_following_rate
    )


def test_cosine_configs_preserve_stage_policies() -> None:
    stage1 = load_stage1_training_config(ROOT / "configs/stage1.yaml")
    stage2 = load_stage2_config(ROOT / "configs/stage2.yaml")
    stage3 = load_stage3_config(ROOT / "configs/stage3.yaml")

    assert stage1.train.checkpoint_policy == "best_lpips"
    assert stage2.train.checkpoint_policy == "best_bpp"
    assert stage2.train.batch_size == 2
    assert stage3.loss.rate_weight == 24.0
    assert not hasattr(stage3.loss, "gan_weight")
