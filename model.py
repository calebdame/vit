"""Model factory helpers for the CIFAR-100 Vision Transformer."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_model(num_classes: int, vit_name: str) -> nn.Module:
    """Construct a ViT-B/16 model with a task-specific classification head."""
    if vit_name =="16":
        model = models.vit_b_16(weights=None)
    else:
        model = models.vit_b_32(weights=None)

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


def to_device(model: nn.Module) -> tuple[nn.Module, torch.device]:
    """Move the model to an available accelerator and return the device."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device
