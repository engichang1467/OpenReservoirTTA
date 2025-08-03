import torch
import torch.nn as nn
import torchvision.models as models

class VGGStyleExtractor(nn.Module):
    def __init__(self, layer='relu3_1', pretrained=True):
        super().__init__()
        vgg = models.vgg19(pretrained=pretrained).features

        # Mapping layer names to indices in torchvision VGG
        self.layer_map = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_1': 10,
            'relu3_3': 14,
            'relu4_1': 19
        }
        assert layer in self.layer_map, f"Unsupported layer {layer}"

        self.slice = nn.Sequential(*[vgg[i] for i in range(self.layer_map[layer] + 1)])

        for param in self.slice.parameters():
            param.requires_grad = False  # Freeze all VGG params

    def forward(self, x):
        with torch.no_grad():
            features = self.slice(x)
        # B × C × H × W → log-variance over H × W
        var = torch.var(features, dim=(2, 3), unbiased=False) + 1e-5
        return torch.log(var)
