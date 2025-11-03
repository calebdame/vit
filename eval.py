import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse, torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from vision_transformer.config import Config
from vision_transformer.utils import (Logger, ensure_dirs, evaluate)
from vision_transformer.data import get_dataloaders, IM_MEAN, IM_STD
from vision_transformer.model import build_model, to_device

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

def save_misclassified_grid(model, test_set, img_size, out_path, device):
    inv = transforms.Normalize(mean=[-m/s for m,s in zip(IM_MEAN,IM_STD)],
                               std=[1/s for s in IM_STD])
    # Re-run predictions using normalized inputs from test_set transforms
    loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    ims, tr, pr = [], [], []
    with torch.no_grad():
        for x,y in loader:
            px = (x - torch.tensor(IM_MEAN).view(3,1,1)) / torch.tensor(IM_STD).view(3,1,1)
            pred = model(px.to(device)).argmax(1).cpu()
            mask = (pred != y)
            for i in torch.nonzero(mask).squeeze():
                ims.append(inv(x[i]).clamp(0,1)); tr.append(y[i].item()); pr.append(pred[i].item())
                if len(ims) >= 36: break
            if len(ims) >= 36: break
    if not ims: return
    n=len(ims); cols=6; rows=(n+cols-1)//cols
    fig = plt.figure(figsize=(cols*2, rows*2))
    for i in range(n):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(ims[i].permute(1,2,0).numpy()); ax.axis("off")
        ax.set_title(f"T:{tr[i]} P:{pr[i]}", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)

def evaluate_test(cfg: Config):
    ensure_dirs(cfg.ckpt_dir, cfg.log_dir, cfg.artifacts_dir)
    logger = Logger(Path(cfg.log_dir) / cfg.test_log)
    # Build loaders; reuse transforms
    _, _, test_loader, test_set, _ = get_dataloaders(cfg)
    model = build_model(cfg.num_classes); model, device = to_device(model)
    ckpt = torch.load(Path(cfg.ckpt_dir) / cfg.best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"]); model.eval()
    loss, acc = evaluate(model, test_loader, torch.nn.CrossEntropyLoss(), device)
    logger.write(f"Test loss:{loss:.4f} Test acc:{acc:.4f}")

    # Confusion matrix + misclassifications
    targets, preds = collect_preds(model, test_loader, device)
    cm = confusion_matrix(targets.numpy(), preds.numpy())
    save_confusion_matrix(cm, Path(cfg.artifacts_dir) / "confusion_matrix.png")
    save_misclassified_grid(model, test_set, cfg.img_size,
                            Path(cfg.artifacts_dir) / "misclassified_grid.png", device)
    logger.write(f"Saved artifacts to {cfg.artifacts_dir}"); logger.close()

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate best checkpoint on CIFAR-100.")
    p.add_argument("--data_dir", type=Path, default=Config.data_dir)
    p.add_argument("--ckpt_dir", type=Path, default=Config.ckpt_dir)
    p.add_argument("--log_dir", type=Path, default=Config.log_dir)
    p.add_argument("--artifacts_dir", type=Path, default=Config.artifacts_dir)
    p.add_argument("--img_size", type=int, default=Config.img_size)
    p.add_argument("--num_workers", type=int, default=Config.num_workers)
    p.add_argument("--batch_size", type=int, default=Config.batch_size)
    return p.parse_args()

if __name__ == "__main__":
    a = parse_args()
    cfg = Config(data_dir=a.data_dir, ckpt_dir=a.ckpt_dir, log_dir=a.log_dir,
                 artifacts_dir=a.artifacts_dir, img_size=a.img_size,
                 num_workers=a.num_workers, batch_size=a.batch_size)
    evaluate_test(cfg)
