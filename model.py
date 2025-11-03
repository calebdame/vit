import torch
from torch import nn
from torchvision import models

def build_model(num_classes: int):
    # ViT-B/16 without pretraining; replace head to 100 classes
    model = models.vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model

def to_device(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device
