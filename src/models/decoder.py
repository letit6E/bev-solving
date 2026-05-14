"""Small 4-down/4-up UNet that all stages share for the BEV decoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):
    def __init__(self, in_c, base_c=32, out_c=1):
        super().__init__()
        c = [base_c, base_c * 2, base_c * 4, base_c * 8]
        self.in_proj = nn.Conv2d(in_c, c[0], 1)
        self.enc1, self.enc2, self.enc3 = _Block(c[0], c[0]), _Block(c[0], c[1]), _Block(c[1], c[2])
        self.bot = _Block(c[2], c[3])
        self.dec3, self.dec2, self.dec1 = _Block(c[3] + c[2], c[2]), _Block(c[2] + c[1], c[1]), _Block(c[1] + c[0], c[0])
        self.out = nn.Conv2d(c[0], out_c, 1)
        self.pool = nn.MaxPool2d(2)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        x = self.in_proj(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bot(self.pool(e3))
        d3 = self.dec3(torch.cat([self._up(b, e3), e3], 1))
        d2 = self.dec2(torch.cat([self._up(d3, e2), e2], 1))
        d1 = self.dec1(torch.cat([self._up(d2, e1), e1], 1))
        return self.out(d1)
