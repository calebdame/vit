from __future__ import annotations

import argparse
import os
import sys

import torch
from pathlib import Path
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler

if __package__ is None or __package__ == "":  # pragma: no cover - runtime path adjustment
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from vit.config import Config
from vit.utils import (
    Logger,
    MetricsRecorder,
    EpochMetrics,
    AverageMeter,
    seed_everything,
    ensure_dirs,
    evaluate,
    dump_hparams,
    log_epoch,
    topk_accuracy,
)
from vit.data import get_dataloaders
from vit.model import build_model, to_device
from vit.optim import make_optimizer, make_scheduler, adjust_lr

def train(cfg: Config) -> None:
    ensure_dirs(cfg.ckpt_dir, cfg.log_dir, cfg.artifacts_dir)
    seed_everything(cfg.seed)

    log_path = Path(cfg.log_dir) / cfg.train_log
    metrics_path = Path(cfg.log_dir) / "training_metrics.jsonl"
    logger = Logger(log_path)
    recorder = MetricsRecorder(metrics_path)

    logger.write("Preparing data loaders…")
    train_loader, val_loader, _, _, _ = get_dataloaders(cfg, logger)

    logger.write("Building model and optimizer…")
    model = build_model(cfg.num_classes)
    model, device = to_device(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, cfg.lr, cfg.weight_decay)

    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = max(1, cfg.warmup_epochs * len(train_loader))
    scheduler = make_scheduler(optimizer, total_steps, warmup_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and cfg.use_amp)

    dump_hparams(logger, cfg)

    best_val_top1 = 0.0
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        train_metrics, global_step = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            cfg.lr,
            warmup_steps,
            global_step,
            epoch,
            cfg.epochs,
            logger,
        )
        log_epoch(logger, recorder, train_metrics)

        val_loss, val_top1, val_top5 = evaluate(model, val_loader, criterion, device)
        metrics = EpochMetrics(
            epoch=epoch,
            stage="val",
            loss=val_loss,
            top1=val_top1,
            top5=val_top5,
            lr=optimizer.param_groups[0]["lr"],
        )
        log_epoch(logger, recorder, metrics)

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            save_path = Path(cfg.ckpt_dir) / cfg.best_ckpt
            torch.save({"model": model.state_dict(), "top1": val_top1}, save_path)
            logger.write(f"Saved new best checkpoint to {save_path} (top1={val_top1:.4f})")

    logger.close()
    recorder.close()


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: _LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    base_lr: float,
    warmup_steps: int,
    global_step: int,
    epoch: int,
    max_epochs: int,
    logger: Logger,
) -> tuple[EpochMetrics, int]:
    """Train for a single epoch and return averaged metrics."""

    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    model.train()
    for step, (images, targets) in enumerate(loader, start=1):
        adjust_lr(optimizer, base_lr, global_step, warmup_steps)
        global_step += 1

        images, targets = images.to(device, non_blocking=True), targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(topk_accuracy(outputs, targets, k=1).item(), batch_size)
        top5_meter.update(topk_accuracy(outputs, targets, k=5).item(), batch_size)

        if step % 25 == 0 or step == len(loader):
            logger.write(
                " | ".join(
                    [
                        f"epoch={epoch:03d}/{max_epochs:03d}",
                        f"step={step:04d}/{len(loader):04d}",
                        f"loss={loss_meter.avg:.4f}",
                        f"top1={top1_meter.avg:.4f}",
                        f"top5={top5_meter.avg:.4f}",
                        f"lr={optimizer.param_groups[0]['lr']:.6f}",
                    ]
                )
            )

    scheduler.step()

    metrics = EpochMetrics(
        epoch=epoch,
        stage="train",
        loss=loss_meter.avg,
        top1=top1_meter.avg,
        top5=top5_meter.avg,
        lr=optimizer.param_groups[0]["lr"],
    )

    return metrics, global_step

def parse_args():
    p = argparse.ArgumentParser(description="Train ViT on CIFAR-100 (from scratch).")
    p.add_argument("--data_dir", type=Path, default=Config.data_dir)
    p.add_argument("--ckpt_dir", type=Path, default=Config.ckpt_dir)
    p.add_argument("--log_dir", type=Path, default=Config.log_dir)
    p.add_argument("--artifacts_dir", type=Path, default=Config.artifacts_dir)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--batch_size", type=int, default=Config.batch_size)
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--weight_decay", type=float, default=Config.weight_decay)
    p.add_argument("--img_size", type=int, default=Config.img_size)
    p.add_argument("--num_workers", type=int, default=Config.num_workers)
    p.add_argument("--warmup_epochs", type=int, default=Config.warmup_epochs)
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = Config(
        data_dir=args.data_dir,
        ckpt_dir=args.ckpt_dir,
        log_dir=args.log_dir,
        artifacts_dir=args.artifacts_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        img_size=args.img_size,
        num_workers=args.num_workers,
        warmup_epochs=args.warmup_epochs,
        use_amp=not args.no_amp,
    )
    train(cfg)
