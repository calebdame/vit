"""Configuration dataclass storing experiment defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Bundle all experiment hyperparameters in a single place."""

    data_dir: Path = Path("./data")
    ckpt_dir: Path = Path("./checkpoints")
    log_dir: Path = Path("./logs")
    artifacts_dir: Path = Path("./artifacts")
    seed: int = 42

    batch_size: int = 128
    epochs: int = 500
    lr: float = 2e-4            # AdamW-friendly for ViTs from scratch
    weight_decay: float = 1e-3   # Strong regularization common in ViT literature
    img_size: int = 224          # ViT-B/16 default
    num_workers: int = 4
    num_classes: int = 100
    warmup_epochs: int = 5
    use_amp: bool = True
    grad_clip_norm: float = 10.0
    label_smoothing: float = 0.01

    # net size
    vit: str = "16"

    # filenames
    best_ckpt: str = "best.pt"
    train_log: str = "training_log.txt"
    test_log: str = "test_eval_log.txt"
