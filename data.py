"""Data loading utilities for the CIFAR-100 Vision Transformer project."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _stratified_split_indices(
    targets: list[int] | torch.Tensor,
    val_ratio: float,
    num_classes: int,
    generator: torch.Generator,
) -> tuple[list[int], list[int]]:
    """Return stratified train/val indices for the given targets."""

    targets_tensor = torch.as_tensor(targets)
    total_samples = targets_tensor.numel()
    desired_val = int(total_samples * val_ratio)

    class_data: dict[int, dict[str, torch.Tensor | float | int]] = {}
    current_val = 0

    for class_idx in range(num_classes):
        class_indices = torch.nonzero(targets_tensor == class_idx, as_tuple=False).squeeze(1)
        if class_indices.numel() == 0:
            continue

        shuffled = class_indices[torch.randperm(class_indices.numel(), generator=generator)]
        exact = float(shuffled.numel() * val_ratio)
        val_count = int(exact)

        class_data[class_idx] = {
            "indices": shuffled,
            "val_count": val_count,
            "remainder": exact - val_count,
        }
        current_val += val_count

    if current_val < desired_val:
        sorted_classes = sorted(class_data.items(), key=lambda item: item[1]["remainder"], reverse=True)
        idx = 0
        while current_val < desired_val and sorted_classes:
            class_idx, data = sorted_classes[idx % len(sorted_classes)]
            if data["val_count"] < data["indices"].numel():
                data["val_count"] += 1
                current_val += 1
            idx += 1
    elif current_val > desired_val:
        sorted_classes = sorted(class_data.items(), key=lambda item: item[1]["remainder"])
        idx = 0
        while current_val > desired_val and sorted_classes:
            class_idx, data = sorted_classes[idx % len(sorted_classes)]
            if data["val_count"] > 0:
                data["val_count"] -= 1
                current_val -= 1
            idx += 1

    train_indices: list[int] = []
    val_indices: list[int] = []
    for data in class_data.values():
        val_count = int(data["val_count"])
        indices = data["indices"]
        val_indices.extend(indices[:val_count].tolist())
        train_indices.extend(indices[val_count:].tolist())

    return train_indices, val_indices

IM_MEAN = (0.485, 0.456, 0.406)
IM_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Return train and evaluation transforms for CIFAR-100."""

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(IM_MEAN, IM_STD),
            # transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=IM_MEAN),
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

    train_transform, eval_transform = build_transforms(cfg.img_size)

    base_train = datasets.CIFAR100(root=cfg.data_dir, train=True, download=True)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_indices, val_indices = _stratified_split_indices(
        base_train.targets,
        val_ratio=0.10,
        num_classes=cfg.num_classes,
        generator=generator,
    )

    train_dataset = datasets.CIFAR100(
        root=cfg.data_dir,
        train=True,
        download=False,
        transform=train_transform,
    )
    val_dataset = datasets.CIFAR100(
        root=cfg.data_dir,
        train=True,
        download=False,
        transform=eval_transform,
    )

    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(val_dataset, val_indices)

    test_set = datasets.CIFAR100(
        root=cfg.data_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )

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
