import torch

def make_optimizer(model, lr, weight_decay):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def make_scheduler(optimizer, total_steps, warmup_steps):
    # Cosine after warmup; smooth start helps with untrained patch/class tokens
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps)
    )

def adjust_lr(optimizer, base_lr, step, warmup_steps):
    if step < warmup_steps:
        scale = base_lr * (step + 1) / max(1, warmup_steps)
        for pg in optimizer.param_groups: pg["lr"] = scale
