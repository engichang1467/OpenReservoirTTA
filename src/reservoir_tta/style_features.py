import torch
from src.models.vgg_style import VGGStyleExtractor

class StyleFeatureExtractor:
    """
    Extracts style vectors (log-variance of VGG features) for a batch of images.
    Intended for use in online clustering during ReservoirTTA.
    """

    def __init__(self, device='cpu', layer='relu3_1'):
        self.device = device
        self.model = VGGStyleExtractor(layer=layer).to(device)
        self.model.eval()

    def extract(self, batch: torch.Tensor) -> torch.Tensor:
        """
        batch: B x 3 x H x W input images, normalized to [0, 1] or standard ImageNet mean/std.
        Returns: B x C log-variance vectors (one per image).
        """
        assert batch.ndim == 4, f"Expected input of shape (B, 3, H, W), got {batch.shape}"
        with torch.no_grad():
            style_vectors = self.model(batch.to(self.device))  # shape: (B, C)
        return style_vectors.cpu()
