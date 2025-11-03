import json, time, random
import torch
from pathlib import Path

class Logger:
    def __init__(self, fpath):
        self.path = Path(fpath)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a", encoding="utf-8")
    def write(self, msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line); self.fh.write(line + "\n"); self.fh.flush()
    def close(self): self.fh.close()

def seed_everything(seed=42):
    random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def accuracy(logits, targets):
    return (logits.argmax(1) == targets).float().mean().item()

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval(); n=0; loss_sum=0.0; acc_sum=0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        out = model(x); loss = criterion(out, y); b = y.size(0)
        loss_sum += loss.item()*b; acc_sum += accuracy(out,y)*b; n += b
    return loss_sum/n, acc_sum/n

def ensure_dirs(*paths):
    for p in paths: Path(p).mkdir(parents=True, exist_ok=True)

def dump_hparams(logger, cfg):
    logger.write(json.dumps({
        "hp": dict(
            batch_size=cfg.batch_size, epochs=cfg.epochs, lr=cfg.lr,
            weight_decay=cfg.weight_decay, img_size=cfg.img_size,
            warmup_epochs=cfg.warmup_epochs
        )
    }))
