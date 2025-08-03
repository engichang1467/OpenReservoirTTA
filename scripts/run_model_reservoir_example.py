import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.resnext import resnext29_8x64d
from src.models.vgg_style import VGGStyleExtractor
from src.reservoir_tta.style_features import StyleFeatureExtractor
from src.reservoir_tta.online_clustering import OnlineStyleClustering
from src.reservoir_tta.model_reservoir import ModelReservoir
from src.tta_methods.eata import EATATestTimeAdaptation

# --- Config ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
TAU = 25.0  # clustering threshold

# --- Normalize like ImageNet ---
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# --- Load dummy CIFAR-100 batch ---
dataset = CIFAR100(root='data/', train=False, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

inputs, labels = next(iter(loader))
inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

# --- Load models ---
backbone = resnext29_8x64d(num_classes=100).to(DEVICE)
style_extractor = StyleFeatureExtractor(device=DEVICE)

# --- Extract style vectors ---
style_vecs = style_extractor.extract(inputs)  # shape: (B, D)

# --- Cluster styles online ---
clustering = OnlineStyleClustering(tau=TAU, device=DEVICE)
assignments = clustering.update(style_vecs)
is_new_cluster = len(clustering) > 1  # simulate discovering a new domain

# --- Set up ModelReservoir with EATA as base method ---
model_reservoir = ModelReservoir(
    base_model=backbone,
    tta_cls=EATATestTimeAdaptation,
    device=DEVICE
)

# --- Maybe add new domain specialist ---
model_reservoir.maybe_add_new_model(inputs, is_new_cluster=is_new_cluster)

# --- Adapt the selected specialist model ---
model_reservoir.adapt(
    inputs=inputs,
    style_vecs=style_vecs,
    centroids=clustering.get_centroids(),
    assignments=assignments
)

# --- Run soft-ensembled prediction ---
preds = model_reservoir.predict(
    inputs=inputs,
    style_vecs=style_vecs,
    centroids=clustering.get_centroids()
)

# --- Output ---
print("True Labels: ", labels.tolist())
print("Predictions: ", preds.tolist())
print(f"# Reservoir Models: {len(model_reservoir.models)}")
print(f"# Domain Clusters: {len(clustering)}")
