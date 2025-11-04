"""Optimizer and scheduler helpers."""

from __future__ import annotations

import torch
from torch import nn


def make_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Create the AdamW optimizer configured for Vision Transformers."""

    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    layer_norm_param_names: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            for param_name, _ in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                layer_norm_param_names.add(full_name)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias") or name in layer_norm_param_names:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    return torch.optim.AdamW(param_groups, lr=lr)


def make_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Return a cosine scheduler that is aware of the warmup period."""

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps)
    )


def adjust_lr(optimizer: torch.optim.Optimizer, base_lr: float, step: int, warmup_steps: int) -> None:
    """Linearly warm up the learning rate for the initial steps."""

    if step < warmup_steps:
        scale = base_lr * (step + 1) / max(1, warmup_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = scale
