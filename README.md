# Vision Transformer — CIFAR-100 Classification

This repository implements a **Vision Transformer (ViT-B/16)** model for image classification on the **CIFAR-100** dataset, trained **from scratch** (no pretraining).  
It uses **PyTorch**, with modular structure, CLI overrides, and full logging.

---

## Overview
This project covers:
1. Dataset prep (CIFAR-100, 90/10 train/val split)
2. Preprocessing (resize 32×32→224×224, augmentations)
3. ViT-B/16 setup (no pretrained weights)
4. Training + checkpointing
5. Evaluation (confusion matrix + misclassified grid)

---

## Repo Structure
```
vit/
├── config.py          # Default config (dataclass)
├── utils.py           # Logging, seeding, metrics
├── data.py            # Data loading & transforms
├── model.py           # ViT-B/16 model builder
├── optim.py           # Optimizer, scheduler, warmup
├── train.py           # CLI entry: training
├── eval.py            # CLI entry: evaluation
└── requirements.txt
```

Generated dirs:
```
checkpoints/   # Saved weights
logs/           # Train/test logs
artifacts/      # Confusion matrix, misclassified images
data/           # Auto-downloaded CIFAR-100
```

---

## Setup
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Train
```
python -m vit.train
```
Override defaults:
```
python -m vit.train --epochs 30 --lr 2e-4 --batch_size 256 --img_size 224 --no_amp
```

Logs → `logs/training_log.txt`  
Checkpoint → `checkpoints/best.pt`

### Evaluate
```
python -m vit.eval
```
Outputs:
- logs/test_eval_log.txt  
- artifacts/confusion_matrix.png  
- artifacts/misclassified_grid.png

---

## 📊 Default Hyperparams
| Param | Default | Note |
|--------|----------|------|
| epochs | 20 | Meets ≥20 epoch requirement |
| lr | 3e-4 | Stable for AdamW + ViT |
| weight_decay | 0.05 | Standard ViT regularization |
| batch_size | 128 | Balanced for GPU efficiency |
| img_size | 224 | ViT-B/16 default |
| warmup_epochs | 2 | Smooth startup |
| optimizer | AdamW | Decoupled regularization |
| scheduler | CosineAnnealingLR | Smooth decay |

---

## Key Design Choices
- **No pretraining:** Required by assignment  
- **AutoAugment(IMAGENET):** Increases patch diversity  
- **Cosine LR + warmup:** Smooth convergence  
- **AMP:** Speeds up training on GPU  
- **Logging:** Plain text for reproducibility  

---

## Code Quality
- Pylint 10/10 compliance  
- Functions ≤30 lines  
- Reproducible seeding

---

## Workflow
```
python -m vit.train --epochs 25
python -m vit.eval
cat logs/training_log.txt
```
