from __future__ import annotations

import torch


def create_fp32_tokenizer_masters(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    checkpoint_weights: dict[str, torch.Tensor] | None,
) -> dict[str, torch.nn.Parameter]:
    parameter_names = {name for name, _ in named_parameters}
    if checkpoint_weights is not None and set(checkpoint_weights) != parameter_names:
        raise ValueError(
            "Tokenizer FP32 master checkpoint names do not match tokenizer parameters"
        )
    masters: dict[str, torch.nn.Parameter] = {}
    for name, parameter in named_parameters:
        value = (
            parameter.detach().float()
            if checkpoint_weights is None
            else checkpoint_weights[name].to(parameter.device, dtype=torch.float32)
        )
        if value.shape != parameter.shape:
            raise ValueError(
                f"Tokenizer FP32 master shape mismatch for {name}: "
                f"{tuple(value.shape)} vs {tuple(parameter.shape)}"
            )
        if checkpoint_weights is not None and not torch.equal(
            value.to(parameter.dtype), parameter
        ):
            raise ValueError(
                f"Tokenizer FP32 master {name} does not reproduce the saved weight"
            )
        masters[name] = torch.nn.Parameter(value.clone())
    return masters


def copy_tokenizer_gradients_to_masters(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    masters: dict[str, torch.nn.Parameter],
) -> None:
    for name, parameter in named_parameters:
        if parameter.grad is None:
            raise RuntimeError(f"Tokenizer parameter {name} received no gradient")
        masters[name].grad = parameter.grad.detach().float()


@torch.no_grad()
def copy_master_weights_to_tokenizer(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    masters: dict[str, torch.nn.Parameter],
) -> None:
    for name, parameter in named_parameters:
        parameter.copy_(masters[name])


def master_checkpoint_state(
    masters: dict[str, torch.nn.Parameter],
) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu() for name, parameter in masters.items()}
