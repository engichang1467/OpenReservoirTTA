import os
import sys
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.resnext import resnext29_8x64d
from src.tta_methods.eata import EATATestTimeAdaptation

# --- Config ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
SEED = 42

# --- ImageNet-style normalization ---
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std =[0.229, 0.224, 0.225]   # ImageNet std
    )
])

# --- Load a small CIFAR-100 test batch ---
dataset = CIFAR100(root='data/', train=False, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

inputs, labels = next(iter(loader))
inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

# --- Load model (untrained or pretrained if you have one) ---
model = resnext29_8x64d(num_classes=100).to(DEVICE)
model.eval()

# --- Run prediction BEFORE adaptation ---
with torch.no_grad():
    logits = model(inputs)
    preds_before = torch.argmax(logits, dim=1)

# --- EATA setup ---
eata = EATATestTimeAdaptation(model=model, steps=1, lr=1e-3, entropy_threshold=1.5, device=DEVICE)

# --- Run adaptation ---
preds_after = eata.adapt(inputs)

# --- Output results ---
print("True Labels:       ", labels.tolist())
print("Predictions Before:", preds_before.tolist())
print("Predictions After: ", preds_after.tolist())
