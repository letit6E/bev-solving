"""Shared image backbones used by multiple model heads."""
import torch.nn as nn
import torchvision


class _ResNet50Backbone(nn.Module):
    """ResNet50 stem + layer1 + layer2, projected to 128 ch (stride 8)."""
    def __init__(self, pretrained=True):
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            except Exception:
                weights = None
        rn = torchvision.models.resnet50(weights=weights)
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool)
        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.proj = nn.Conv2d(512, 128, 1)

    def forward(self, x):
        return self.proj(self.layer2(self.layer1(self.stem(x))))
