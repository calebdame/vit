"""Data loading utilities for the CIFAR-100 Vision Transformer project."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

IM_MEAN = (0.485, 0.456, 0.406)
IM_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Return train and evaluation transforms for CIFAR-100."""

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(IM_MEAN, IM_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(img_size, antialias=True),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IM_MEAN, IM_STD),
        ]
    )
    return train_transform, eval_transform


def get_dataloaders(cfg, logger=None):
    """Return train/val/test dataloaders along with the raw test dataset."""

    base_train = datasets.CIFAR100(root=cfg.data_dir, train=True, download=True)
    test_set = datasets.CIFAR100(root=cfg.data_dir, train=False, download=True)

    n_total = len(base_train)
    n_val = int(0.10 * n_total)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = random_split(base_train, [n_train, n_val], generator=generator)

    train_transform, eval_transform = build_transforms(cfg.img_size)
    train_set.dataset.transform = train_transform
    val_set.dataset.transform = eval_transform
    test_set.transform = eval_transform

    loader_kwargs = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    if logger is not None:
        logger.write(
            f"Dataset split — train:{len(train_set)} val:{len(val_set)} test:{len(test_set)}"
        )
        logger.write(
            " | ".join(
                [
                    f"steps/train={len(train_loader)}",
                    f"steps/val={len(val_loader)}",
                    f"steps/test={len(test_loader)}",
                ]
            )
        )

    return train_loader, val_loader, test_loader, test_set, (train_transform, eval_transform)
