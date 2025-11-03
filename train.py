import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse, json, torch
from pathlib import Path
from torch import nn
from vit.config import Config
from vit.utils import (Logger, seed_everything, ensure_dirs,
                                     evaluate, dump_hparams, accuracy)
from vit.data import get_dataloaders
from vit.model import build_model, to_device
from vit.optim import make_optimizer, make_scheduler, adjust_lr

def train(cfg: Config):
    ensure_dirs(cfg.ckpt_dir, cfg.log_dir, cfg.artifacts_dir)
    seed_everything(cfg.seed)
    logger = Logger(Path(cfg.log_dir) / cfg.train_log)

    train_loader, val_loader, _, _, _ = get_dataloaders(cfg, logger)
    model = build_model(cfg.num_classes); model, device = to_device(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, cfg.lr, cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = max(1, cfg.warmup_epochs * len(train_loader))
    scheduler = make_scheduler(optimizer, total_steps, warmup_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and cfg.use_amp))

    dump_hparams(logger, cfg)
    best_val_acc, global_step = 0.0, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train(); ep_loss=0.0; ep_acc=0.0; n_seen=0
        for images, targets in train_loader:
            adjust_lr(optimizer, cfg.lr, global_step, warmup_steps); global_step += 1
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                outputs = model(images); loss = criterion(outputs, targets)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            b = targets.size(0); ep_loss += loss.item()*b; ep_acc += accuracy(outputs,targets)*b; n_seen += b
        scheduler.step()
        tr_loss, tr_acc = ep_loss/n_seen, ep_acc/n_seen
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "acc": val_acc},
                       Path(cfg.ckpt_dir) / cfg.best_ckpt)
        logger.write(f"Epoch {epoch:02d}/{cfg.epochs} "
                     f"train_loss:{tr_loss:.4f} train_acc:{tr_acc:.4f} "
                     f"val_loss:{val_loss:.4f} val_acc:{val_acc:.4f} "
                     f"best_val_acc:{best_val_acc:.4f}")
    logger.close()

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
        data_dir=args.data_dir, ckpt_dir=args.ckpt_dir, log_dir=args.log_dir,
        artifacts_dir=args.artifacts_dir, seed=args.seed, batch_size=args.batch_size,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        img_size=args.img_size, num_workers=args.num_workers,
        warmup_epochs=args.warmup_epochs, use_amp=not args.no_amp
    )
    train(cfg)
