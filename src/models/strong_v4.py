"""StrongBEVEncoderDecoder: BEV trunk with ASPP and 2-stage U-Net used by v7+.

Lives separately from `decoder.SmallUNet` because it's deeper and uses GN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.lss import ASPP2d, ConvGNAct, ResidualBlock2d


class StrongBEVEncoderDecoder(nn.Module):
    def __init__(self, in_c, base_c=96, groups=8):
        super().__init__()
        self.stem = nn.Sequential(
            ConvGNAct(in_c, base_c, k=3, s=1, p=1, groups=groups, act=True),
            ResidualBlock2d(base_c, base_c, stride=1, groups=groups),
        )
        self.down1 = nn.Sequential(
            ResidualBlock2d(base_c, base_c * 2, stride=2, groups=groups),
            ResidualBlock2d(base_c * 2, base_c * 2, stride=1, groups=groups),
        )
        self.down2 = nn.Sequential(
            ResidualBlock2d(base_c * 2, base_c * 4, stride=2, groups=groups),
            ResidualBlock2d(base_c * 4, base_c * 4, stride=1, groups=groups),
        )
        self.aspp = ASPP2d(base_c * 4, base_c * 4, rates=(1, 3, 6), groups=groups)
        self.up1 = nn.Sequential(
            ConvGNAct(base_c * 4 + base_c * 2, base_c * 2, k=3, s=1, p=1, groups=groups, act=True),
            ResidualBlock2d(base_c * 2, base_c * 2, stride=1, groups=groups),
        )
        self.up0 = nn.Sequential(
            ConvGNAct(base_c * 2 + base_c, base_c, k=3, s=1, p=1, groups=groups, act=True),
            ResidualBlock2d(base_c, base_c, stride=1, groups=groups),
        )
        self.head = nn.Conv2d(base_c, 1, 1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        b = self.aspp(s2)
        u1 = self.up1(torch.cat([F.interpolate(b, size=s1.shape[-2:], mode="bilinear", align_corners=False), s1], dim=1))
        u0 = self.up0(torch.cat([F.interpolate(u1, size=s0.shape[-2:], mode="bilinear", align_corners=False), s0], dim=1))
        return self.head(u0)
