"""v1: ResNet-18 layer2 backbone, parameter-free voxel projection, SmallUNet.

Phase 1 baseline. ~3M trainable params. Trained from ImageNet init.
"""
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from src.models.voxel import VoxelBEVHead


class _R18Stem(nn.Module):
    def __init__(self):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2)

    def forward(self, x):
        return self.backbone(x)


class MultiCamBEV(VoxelBEVHead):
    def __init__(self, freeze_backbone=False):
        super().__init__(feat_dim=128, proj_dim=64)
        self.backbone = _R18Stem()
        self._frozen = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if mode and self._frozen:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self
