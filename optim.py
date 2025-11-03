"""Optimizer and scheduler helpers."""

from __future__ import annotations

import torch


def make_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Create the AdamW optimizer configured for Vision Transformers."""

    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


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
