import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import List

class ModelReservoir:
    """
    Manages a reservoir of domain-specialized models.
    Handles model selection, cloning, adaptation, and ensemble prediction.
    """

    def __init__(self, base_model: nn.Module, tta_cls, device='cpu'):
        """
        base_model: initial pretrained backbone (shared across all domains).
        tta_cls: class of the base TTA method (e.g., EATATestTimeAdaptation)
        """
        self.device = device
        self.tta_cls = tta_cls
        self.base_model = base_model.to(device).eval()

        # Start with one model in the reservoir (source model)
        self.models: List[nn.Module] = [deepcopy(self.base_model)]
        self.tta_wrappers: List[object] = [tta_cls(self.models[0], device=device)]

    def _clone_model(self, reference_model: nn.Module) -> nn.Module:
        """Clone weights from a reference model to a new model."""
        new_model = deepcopy(reference_model)
        return new_model.to(self.device)

    def _confidence_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        return -torch.sum(probs * torch.log(probs + 1e-6), dim=1)

    def _select_confident_model(self, inputs: torch.Tensor) -> int:
        """
        Heuristic: choose the model with lowest entropy on average.
        """
        min_entropy = float('inf')
        best_idx = 0
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                logits = model(inputs)
                entropy = self._confidence_entropy(logits).mean().item()
            if entropy < min_entropy:
                min_entropy = entropy
                best_idx = i
        return best_idx

    def maybe_add_new_model(self, inputs: torch.Tensor, is_new_cluster: bool):
        """
        Called after clustering to conditionally create a new domain model.
        inputs: current test batch
        is_new_cluster: True if clustering says this domain is novel
        """
        if is_new_cluster:
            ref_idx = self._select_confident_model(inputs)
            new_model = self._clone_model(self.models[ref_idx])
            self.models.append(new_model)
            self.tta_wrappers.append(self.tta_cls(new_model, device=self.device))

    def _soft_assign_weights(self, style_vec: torch.Tensor, centroids: torch.Tensor, temperature=1.0):
        """
        Compute softmax over negative distances → higher weight = closer to centroid.
        Returns a (K,) vector of weights summing to 1.
        """
        dists = torch.norm(centroids - style_vec.unsqueeze(0), dim=1)  # (K,)
        weights = F.softmax(-dists / temperature, dim=0)  # closer → higher weight
        return weights

    def adapt(self, inputs: torch.Tensor, style_vecs: torch.Tensor, centroids: torch.Tensor, assignments: List[int]):
        """
        Adapt the most relevant specialist model for this batch (θk*).
        Updates only that model using the base TTA method.
        """
        B = inputs.size(0)
        device = self.device

        # Select the dominant assignment in the batch (can later be soft-assigned per sample)
        dominant_cluster = max(set(assignments), key=assignments.count)
        specialist_model = self.tta_wrappers[dominant_cluster]
        _ = specialist_model.adapt(inputs)  # update only this model

    def predict(self, inputs: torch.Tensor, style_vecs: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with soft-ensembling of model outputs using assignment weights.
        No adaptation here — just weighted prediction.
        """
        ensemble_logits = torch.zeros(inputs.size(0), self.base_model.linear.out_features).to(self.device)

        for i in range(inputs.size(0)):
            x_i = inputs[i].unsqueeze(0)
            style_i = style_vecs[i]
            weights = self._soft_assign_weights(style_i, centroids)  # (K,)

            logits_list = []
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    logits = model(x_i)
                logits_list.append(logits.squeeze(0))  # shape: (C,)

            stacked_logits = torch.stack(logits_list, dim=0)  # (K, C)
            weighted_logits = torch.sum(weights.view(-1, 1) * stacked_logits, dim=0)
            ensemble_logits[i] = weighted_logits

        return torch.argmax(ensemble_logits, dim=1)  # Final prediction
