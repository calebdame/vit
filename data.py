from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch

IM_MEAN, IM_STD = (0.485,0.456,0.406), (0.229,0.224,0.225)

def build_transforms(img_size):
    train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(IM_MEAN, IM_STD),
    ])
    evalt = transforms.Compose([
        transforms.Resize(img_size, antialias=True),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IM_MEAN, IM_STD),
    ])
    return train, evalt

def get_dataloaders(cfg, logger=None):
    base_train = datasets.CIFAR100(root=cfg.data_dir, train=True, download=True)
    test_set = datasets.CIFAR100(root=cfg.data_dir, train=False, download=True)
    n_total = len(base_train); n_val = int(0.10 * n_total); n_train = n_total - n_val
    g = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = random_split(base_train, [n_train, n_val], generator=g)
    t_train, t_eval = build_transforms(cfg.img_size)
    train_set.dataset.transform = t_train; val_set.dataset.transform = t_eval
    test_set.transform = t_eval

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)
    if logger:
        logger.write(f"Split train:{len(train_set)} val:{len(val_set)} test:{len(test_set)}")
        logger.write(f"Steps/epoch train:{len(train_loader)} val:{len(val_loader)} test:{len(test_loader)}")
    return train_loader, val_loader, test_loader, test_set, (t_train, t_eval)
