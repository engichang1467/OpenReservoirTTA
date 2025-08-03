import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.resnext import resnext29_8x64d
from src.datasets.cifar100c_loader import get_all_cifar100c_corruptions
from src.datasets import get_cifar100c_dataset
from src.reservoir_tta.style_features import StyleFeatureExtractor
from src.reservoir_tta.online_clustering import OnlineStyleClustering
from src.reservoir_tta.model_reservoir import ModelReservoir
from src.tta_methods.eata import EATATestTimeAdaptation

# --- Load config ---
with open('config/default_config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')

# --- Transforms (ImageNet normalization) ---
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# --- Dataset ---
dataset = CIFAR100(root='data/', train=False, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False)

# --- Initialize components ---
backbone = resnext29_8x64d(num_classes=cfg['num_classes']).to(device)
style_extractor = StyleFeatureExtractor(device=device)
clustering = OnlineStyleClustering(tau=cfg['tau'], max_reservoir_size=cfg['max_reservoir_size'], device=device)
model_reservoir = ModelReservoir(
    base_model=backbone,
    tta_cls=lambda model, device: EATATestTimeAdaptation(
        model, steps=cfg['tta_steps'], lr=cfg['tta_lr'],
        entropy_threshold=cfg['entropy_threshold'], device=device
    ),
    device=device
)

# --- Evaluation loop ---
correct = 0
total = 0

# for inputs, labels in loader:
#     inputs, labels = inputs.to(device), labels.to(device)

#     # Step 1: Extract style vectors
#     print("Step 1: Extract style vectors")
#     style_vecs = style_extractor.extract(inputs)

#     # Step 2: Update clustering
#     print("Step 2: Extract style vectors")
#     old_cluster_count = len(clustering)
#     assignments = clustering.update(style_vecs)
#     is_new_cluster = len(clustering) > old_cluster_count

#     # Step 3: Possibly add new domain-specialist model
#     print("Step 3: Possibly add new domain-specialist model")
#     model_reservoir.maybe_add_new_model(inputs, is_new_cluster)

#     # Step 4: Adapt the specialist model
#     print("Step 4: Adapt the specialist model")
#     centroids = clustering.get_centroids()
#     model_reservoir.adapt(inputs, style_vecs, centroids, assignments)

#     # Step 5: Predict using ensemble
#     print("Step 5: Predict using ensemble")
#     preds = model_reservoir.predict(inputs, style_vecs, centroids)

#     # Track accuracy
#     correct += (preds == labels).sum().item()
#     total += labels.size(0)

os.makedirs("outputs", exist_ok=True)
cluster_accuracy_log = defaultdict(list)
style_cache = []

for corruption in get_all_cifar100c_corruptions():
    print(f"\nEvaluating corruption: {corruption}")

    dataset = get_cifar100c_dataset(
        data_dir='data/',
        corruption=corruption,
        severity=3,
        transform=transform
    )
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False)

    # Reset state for each corruption if needed
    correct = 0
    total = 0
    clustering.reset()
    model_reservoir = ModelReservoir(
        base_model=backbone,
        tta_cls=lambda model, device: EATATestTimeAdaptation(
            model, steps=cfg['tta_steps'], lr=cfg['tta_lr'],
            entropy_threshold=cfg['entropy_threshold'], device=device
        ),
        device=device
    )

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        style_vecs = style_extractor.extract(inputs)
        assignments = clustering.update(style_vecs)
        is_new_cluster = len(clustering) > len(set(assignments))

        model_reservoir.maybe_add_new_model(inputs, is_new_cluster)
        model_reservoir.adapt(inputs, style_vecs, clustering.get_centroids(), assignments)
        preds = model_reservoir.predict(inputs, style_vecs, clustering.get_centroids())

        # Accuracy
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Track per-cluster accuracy
        for pred, true, cluster_id in zip(preds, labels, assignments):
            cluster_accuracy_log[cluster_id].append(int(pred == true))

        # Cache for t-SNE
        for v, cluster_id in zip(style_vecs, assignments):
            style_cache.append((v.cpu().numpy(), cluster_id))

    acc = 100 * correct / total
    print(f"[{corruption}] Accuracy: {acc:.2f}% | Domains: {len(clustering)} | Specialists: {len(model_reservoir.models)}")


# # --- Report results ---
# acc = 100 * correct / total
# print(f"\nReservoirTTA Accuracy on CIFAR-100 test set: {acc:.2f}%")
# print(f"Total domain clusters: {len(clustering)}")
# print(f"Total models in reservoir: {len(model_reservoir.models)}")


with open("outputs/cluster_accuracy.txt", "w") as f:
    for cid in sorted(cluster_accuracy_log):
        acc = np.mean(cluster_accuracy_log[cid])
        f.write(f"Cluster {cid}: Accuracy = {acc * 100:.2f}%\n")


features, cluster_ids = zip(*style_cache)
features = np.stack(features)
cluster_ids = np.array(cluster_ids)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
proj = tsne.fit_transform(features)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(proj[:, 0], proj[:, 1], c=cluster_ids, cmap='tab20', s=10)
plt.colorbar(scatter, label='Cluster ID')
plt.title("t-SNE of Style Vectors Across Corruptions")
plt.savefig("outputs/style_tsne.png", dpi=300)
