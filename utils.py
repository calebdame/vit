"""Utility helpers for the Vision Transformer training pipeline."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch


class Logger:
    """Lightweight file logger that mirrors writes to stdout."""

    def __init__(self, fpath: Path | str):
        self.path = Path(fpath)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class MetricsRecorder:
    """Append-only JSONL metrics logger for programmatic analysis."""

    def __init__(self, fpath: Path | str):
        self.path = Path(fpath)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, payload: Dict[str, float | int | str]) -> None:
        enriched = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **payload,
        }
        self._fh.write(json.dumps(enriched) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


@dataclass
class EpochMetrics:
    """Container holding metrics for a single epoch/stage pair."""

    epoch: int
    stage: str
    loss: float
    top1: float
    top5: float
    lr: float


class AverageMeter:
    """Maintains the running average of streaming values."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.value = value
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # Guard in CPU-only CI.
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Return top-k accuracy for the provided logits."""

    k = min(k, logits.size(1))
    _, pred = logits.topk(k, dim=1)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    # A sample counts as correct if any of the top-k predictions match the target.
    return correct.any(dim=1).float().mean()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: Iterable, criterion: torch.nn.Module, device: torch.device) -> Tuple[float, float, float]:
    """Evaluate a model and return loss, top-1, and top-5 accuracy."""

    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(topk_accuracy(outputs, targets, k=1).item(), batch_size)
        top5_meter.update(topk_accuracy(outputs, targets, k=5).item(), batch_size)

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


def ensure_dirs(*paths: Path | str) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def dump_hparams(logger: Logger, cfg) -> None:
    payload = {
        "hyperparameters": {
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "img_size": cfg.img_size,
            "warmup_epochs": cfg.warmup_epochs,
        }
    }
    logger.write(json.dumps(payload))


def log_epoch(logger: Logger, recorder: MetricsRecorder, metrics: EpochMetrics) -> None:
    logger.write(
        " | ".join(
            [
                f"epoch={metrics.epoch:03d}",
                f"stage={metrics.stage}",
                f"loss={metrics.loss:.4f}",
                f"top1={metrics.top1:.4f}",
                f"top5={metrics.top5:.4f}",
                f"lr={metrics.lr:.6f}",
            ]
        )
    )
    recorder.log(asdict(metrics))
