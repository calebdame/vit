"""Model factory helpers for the CIFAR-100 Vision Transformer."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_model(
    num_classes: int,
    vit_name: str,
    head_dropout: float,
    stochastic_depth_prob: float,
) -> nn.Module:
    """Construct a ViT model with regularised classification head."""

    vit_kwargs = dict(weights=None, stochastic_depth_prob=stochastic_depth_prob)
    if vit_name == "16":
        model = models.vit_b_16(**vit_kwargs)
    else:
        model = models.vit_b_32(**vit_kwargs)

    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(p=head_dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def to_device(model: nn.Module) -> tuple[nn.Module, torch.device]:
    """Move the model to an available accelerator and return the device."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device
