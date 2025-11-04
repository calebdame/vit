"""Model factory helpers for the CIFAR-100 Vision Transformer."""

from __future__ import annotations

import inspect
import warnings

import torch
from torch import nn
from torchvision import models


def _build_vit(vit_name: str, stochastic_depth_prob: float) -> nn.Module:
    """Instantiate a torchvision ViT, handling version-specific kwargs.

    Older versions of torchvision (<0.14) do not accept ``stochastic_depth_prob``.
    We only pass the argument when it is supported to avoid ``TypeError``.
    """

    vit_kwargs = {"weights": None}

    builder = models.vit_b_16 if vit_name == "16" else models.vit_b_32
    sig = inspect.signature(builder)
    if "stochastic_depth_prob" in sig.parameters:
        vit_kwargs["stochastic_depth_prob"] = stochastic_depth_prob
    elif stochastic_depth_prob:
        warnings.warn(
            "stochastic_depth_prob is not supported by this torchvision version; "
            "proceeding without stochastic depth.",
            RuntimeWarning,
            stacklevel=2,
        )

    return builder(**vit_kwargs)


def build_model(
    num_classes: int,
    vit_name: str,
    head_dropout: float,
    stochastic_depth_prob: float,
) -> nn.Module:
    """Construct a ViT model with regularised classification head."""

    model = _build_vit(vit_name, stochastic_depth_prob)

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
