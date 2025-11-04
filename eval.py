from __future__ import annotations

import argparse
import os
import sys

import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix
from torchvision import transforms

if __package__ is None or __package__ == "":  # pragma: no cover - runtime path adjustment
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from vit.config import Config
from vit.utils import Logger, ensure_dirs, evaluate, MetricsRecorder, EpochMetrics, log_epoch
from vit.data import get_dataloaders, IM_MEAN, IM_STD
from vit.model import build_model, to_device

def save_confusion_matrix(cm, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig)

@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval(); t_all=[]; p_all=[]
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        p = model(x).argmax(1)
        t_all.append(y.cpu()); p_all.append(p.cpu())
    return torch.cat(t_all), torch.cat(p_all)

def save_misclassified_grid(
    model: torch.nn.Module,
    loader,
    out_path: Path,
    device: torch.device,
    limit: int = 36,
) -> None:
    """Render a grid of misclassified examples for qualitative analysis."""

    inverse_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(IM_MEAN, IM_STD)],
        std=[1 / s for s in IM_STD],
    )

    images, targets, predictions = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            preds = model(inputs).argmax(1).cpu()
            mismatch = preds.ne(labels)
            inputs_cpu = inputs.detach().cpu()
            if mismatch.any():
                for idx in torch.nonzero(mismatch, as_tuple=False).flatten():
                    if len(images) >= limit:
                        break
                    denorm = inverse_normalize(inputs_cpu[idx].clone()).clamp(0, 1)
                    images.append(denorm)
                    targets.append(labels[idx].item())
                    predictions.append(preds[idx].item())
            if len(images) >= limit:
                break

    if not images:
        return

    n_images = len(images)
    cols = 6
    rows = (n_images + cols - 1) // cols
    figure = plt.figure(figsize=(cols * 2, rows * 2))
    for index, image in enumerate(images):
        axis = figure.add_subplot(rows, cols, index + 1)
        axis.imshow(image.permute(1, 2, 0).numpy())
        axis.axis("off")
        axis.set_title(f"T:{targets[index]} P:{predictions[index]}", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)

def evaluate_test(cfg: Config) -> None:
    ensure_dirs(cfg.ckpt_dir, cfg.log_dir, cfg.artifacts_dir)
    logger = Logger(Path(cfg.log_dir) / cfg.test_log)
    recorder = MetricsRecorder(Path(cfg.log_dir) / "evaluation_metrics.jsonl")

    logger.write("Loading datasets and checkpoint…")
    _, _, test_loader, _, _ = get_dataloaders(cfg)
    model = build_model(
        num_classes=cfg.num_classes,
        vit_name=cfg.vit,
        head_dropout=cfg.head_dropout,
        stochastic_depth_prob=cfg.stochastic_depth_prob,
    )
    model, device = to_device(model)
    ckpt_path = Path(cfg.ckpt_dir) / cfg.best_ckpt
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_top1, test_top5 = evaluate(model, test_loader, criterion, device)
    metrics = EpochMetrics(
        epoch=0,
        stage="test",
        loss=test_loss,
        top1=test_top1,
        top5=test_top5,
        lr=0.0,
    )
    log_epoch(logger, recorder, metrics)

    targets, preds = collect_preds(model, test_loader, device)
    cm = confusion_matrix(targets.numpy(), preds.numpy())
    save_confusion_matrix(cm, Path(cfg.artifacts_dir) / "confusion_matrix.png")
    save_misclassified_grid(
        model,
        test_loader,
        Path(cfg.artifacts_dir) / "misclassified_grid.png",
        device,
    )
    logger.write(f"Saved artifacts to {cfg.artifacts_dir}")

    logger.close()
    recorder.close()

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate best checkpoint on CIFAR-100.")
    p.add_argument("--data_dir", type=Path, default=Config.data_dir)
    p.add_argument("--ckpt_dir", type=Path, default=Config.ckpt_dir)
    p.add_argument("--log_dir", type=Path, default=Config.log_dir)
    p.add_argument("--artifacts_dir", type=Path, default=Config.artifacts_dir)
    p.add_argument("--img_size", type=int, default=Config.img_size)
    p.add_argument("--num_workers", type=int, default=Config.num_workers)
    p.add_argument("--batch_size", type=int, default=Config.batch_size)
    p.add_argument("--vit", type=str, default=Config.vit)
    p.add_argument("--head_dropout", type=float, default=Config.head_dropout)
    p.add_argument(
        "--stochastic_depth_prob",
        type=float,
        default=Config.stochastic_depth_prob,
    )
    return p.parse_args()

if __name__ == "__main__":
    a = parse_args()
    cfg = Config(
        data_dir=a.data_dir,
        ckpt_dir=a.ckpt_dir,
        log_dir=a.log_dir,
        artifacts_dir=a.artifacts_dir,
        img_size=a.img_size,
        num_workers=a.num_workers,
        batch_size=a.batch_size,
        vit=a.vit,
        head_dropout=a.head_dropout,
        stochastic_depth_prob=a.stochastic_depth_prob,
    )
    evaluate_test(cfg)
