import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class EATATestTimeAdaptation:
    """
    Simplified EATA-style test-time adaptation for classification.
    Only adapts batch norm affine parameters using unlabeled entropy loss.
    """

    def __init__(self, model: nn.Module, steps=1, lr=1e-3, entropy_threshold=1.5, device='cpu'):
        self.model = model.to(device)
        self.steps = steps
        self.lr = lr
        self.entropy_threshold = entropy_threshold
        self.device = device

        self._setup_optimizer()

    def _setup_optimizer(self):
        # Only update BN affine parameters (weight & bias)
        params = []
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    params.append(m.weight)
                if m.bias is not None:
                    params.append(m.bias)
        self.optimizer = torch.optim.SGD(params, lr=self.lr)

    def _entropy(self, p):
        return -torch.sum(p * torch.log(p + 1e-6), dim=1)

    def adapt(self, inputs: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        One adaptation loop on a batch of inputs.
        Returns: predictions or logits.
        """
        self.model.train()

        for _ in range(self.steps):
            outputs = self.model(inputs)
            probs = F.softmax(outputs, dim=1)

            # EATA-style sample filtering
            entropy = self._entropy(probs)
            mask = entropy < self.entropy_threshold
            if mask.sum() == 0:
                break  # no confident samples

            loss = entropy[mask].mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            logits = self.model(inputs)
            return logits if return_logits else torch.argmax(logits, dim=1)
