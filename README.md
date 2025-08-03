# OpenReservoirTTA

**OpenReservoirTTA** is a simplified, extensible reimplementation of [ReservoirTTA (arXiv 2505.14511)](https://arxiv.org/abs/2505.14511), a plug-in test-time adaptation framework for non-stationary, evolving, and recurring domains. This implementation supports per-corruption evaluation, per-domain logging, and t-SNE visualization.

---

## 🔍 Motivation

Standard test-time adaptation (TTA) methods update a shared model incrementally, which leads to:
- **Catastrophic forgetting** over long-term domain shifts
- **Adaptation collapse** on recurring domains
- **Overwriting between domains**

**ReservoirTTA** avoids these by:
- Maintaining a reservoir of domain-specific models
- Identifying new and recurring domains online using style features
- Adapting each model independently and routing test samples accordingly

---

## 📁 Project Structure

```

OpenReservoirTTA/
├── config/
│   └── default_config.yaml          # Hyperparameters (τ, lr, steps, etc.)
├── scripts/
│   ├── train_eval.py                # Main entry point for ReservoirTTA pipeline
│   └── run_eata_example.py          # Standalone EATA demo on CIFAR-100
├── src/
│   ├── datasets/                    # CIFAR-100-C loader and corruption tools
│   ├── models/                      # Backbone (ResNeXt-29), VGG-style extractor
│   ├── tta_methods/                 # Simplified EATA for unsupervised TTA
│   └── reservoir_tta/              # Core ReservoirTTA logic
│       ├── style_features.py
│       ├── online_clustering.py
│       ├── model_reservoir.py
│       └── reservoir_sampling.py
├── outputs/                         # t-SNE plots, logs, and evaluation results
└── README.md

````

---

## ⚙️ Installation

```bash
git clone https://github.com/engichang1467/OpenReservoirTTA.git
cd OpenReservoirTTA

# (Optional) Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
````

---

## 🚀 Getting Started

### 🔁 Run Full ReservoirTTA Pipeline

```bash
python scripts/train_eval.py
```

* Evaluates on all 19 CIFAR-100-C corruption types (severity 3)
* Logs accuracy per corruption and per discovered domain
* Saves a 2D t-SNE visualization of style space

### ⚗️ Test the EATA Baseline

```bash
python scripts/run_eata_example.py
```

---

## 📊 Outputs

| Output File                    | Description                          |
| ------------------------------ | ------------------------------------ |
| `outputs/cluster_accuracy.txt` | Per-domain cluster test accuracy     |
| `outputs/style_tsne.png`       | t-SNE of all test-time style vectors |
| `stdout`                       | Accuracy logs for each corruption    |

---

## 🧠 Core Concepts

| Component                | Description                                       |
| ------------------------ | ------------------------------------------------- |
| `StyleFeatureExtractor`  | Uses VGG-19 to compute log-variance style vectors |
| `OnlineStyleClustering`  | DP-Means–like clustering of test-time domains     |
| `ModelReservoir`         | Manages domain-specialist models and adaptation   |
| `EATATestTimeAdaptation` | BN-only entropy minimization adaptation loop      |

---

## 📚 Reference

> Vray, G., Tomar, D., Gao, X., Thiran, J.-P., Shelhamer, E., & Bozorgtabar, B. (2025).
> **ReservoirTTA: Prolonged Test-Time Adaptation for Evolving and Recurring Domains**
> *arXiv preprint arXiv:2505.14511*
> [\[Paper\]](https://arxiv.org/abs/2505.14511)

---

## 📝 License

MIT License. See `LICENSE` file for details.

---

## 🙋‍♀️ Acknowledgments

* Inspired by the official ReservoirTTA paper (EPFL, UBC, Vector Institute)
* Built using PyTorch and torchvision

