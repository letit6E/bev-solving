"""CSPNeXt (RTMDet) backbone + FPN, lifted out of stage_v7 notebooks.

The pretrained backbone key remapping handles MMDetection-style RTMDet
`backbone.<stage>.conv2.depthwise_conv/pointwise_conv` -> our nn.Sequential keys.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.lss import ConvGNAct


class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, k, stride=1, padding=None, groups=1,
                 bias=False, eps=1e-3, momentum=0.01, act=True):
        super().__init__()
        if padding is None:
            padding = k // 2
        self.conv = nn.Conv2d(in_c, out_c, k, stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_c, eps=eps, momentum=momentum)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            a = self.pool(x.float())
            a = self.fc(a)
            a = self.act(a)
        return x * a.to(dtype=x.dtype)


class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), eps=1e-3, momentum=0.01):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = ConvBNAct(in_channels, mid_channels, 1, eps=eps, momentum=momentum)
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes
        ])
        self.conv2 = ConvBNAct(mid_channels * (len(kernel_sizes) + 1), out_channels, 1, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.conv1(x)
        with torch.cuda.amp.autocast(enabled=False):
            x32 = x.float()
            x = torch.cat([x32] + [pool(x32) for pool in self.poolings], dim=1)
        x = x.to(dtype=self.conv2.conv.weight.dtype)
        return self.conv2(x)


class CSPNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=0.5, add_identity=True, kernel_size=5,
                 eps=1e-3, momentum=0.01):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBNAct(in_channels, hidden_channels, 3, eps=eps, momentum=momentum)
        self.conv2 = nn.Sequential(
            ConvBNAct(hidden_channels, hidden_channels, kernel_size, groups=hidden_channels, eps=eps, momentum=momentum),
            ConvBNAct(hidden_channels, out_channels, 1, eps=eps, momentum=momentum),
        )
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return y + x if self.add_identity else y


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=0.5, num_blocks=1,
                 add_identity=True, channel_attention=True, eps=1e-3, momentum=0.01):
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvBNAct(in_channels, mid_channels, 1, eps=eps, momentum=momentum)
        self.short_conv = ConvBNAct(in_channels, mid_channels, 1, eps=eps, momentum=momentum)
        self.final_conv = ConvBNAct(2 * mid_channels, out_channels, 1, eps=eps, momentum=momentum)
        self.blocks = nn.Sequential(*[
            CSPNeXtBlock(mid_channels, mid_channels, expansion=1.0, add_identity=add_identity, eps=eps, momentum=momentum)
            for _ in range(num_blocks)
        ])
        self.attention = ChannelAttention(2 * mid_channels) if channel_attention else nn.Identity()

    def forward(self, x):
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)
        x = torch.cat((x_main, x_short), dim=1)
        x = self.attention(x)
        return self.final_conv(x)


class CSPNeXtBackboneFromRTMDet(nn.Module):
    arch_settings = {
        "P5": [
            [64, 128, 3, True, False],
            [128, 256, 6, True, False],
            [256, 512, 6, True, False],
            [512, 1024, 3, False, True],
        ]
    }

    def __init__(self, arch="P5", deepen_factor=1.0, widen_factor=1.0,
                 expand_ratio=0.5, channel_attention=True,
                 out_indices=(2, 3, 4), eps=1e-3, momentum=0.01):
        super().__init__()
        arch_setting = self.arch_settings[arch]
        self.out_indices = out_indices
        c0 = int(arch_setting[0][0] * widen_factor)
        self.stem = nn.Sequential(
            ConvBNAct(3, c0 // 2, 3, stride=2, eps=eps, momentum=momentum),
            ConvBNAct(c0 // 2, c0 // 2, 3, stride=1, eps=eps, momentum=momentum),
            ConvBNAct(c0 // 2, c0, 3, stride=1, eps=eps, momentum=momentum),
        )
        self.layers = ["stem"]
        for i, (in_c, out_c, num_blocks, add_identity, use_spp) in enumerate(arch_setting):
            in_c = int(in_c * widen_factor)
            out_c = int(out_c * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = [ConvBNAct(in_c, out_c, 3, stride=2, eps=eps, momentum=momentum)]
            if use_spp:
                stage.append(SPPBottleneck(out_c, out_c, eps=eps, momentum=momentum))
            stage.append(CSPLayer(
                out_c, out_c,
                expand_ratio=expand_ratio,
                num_blocks=num_blocks,
                add_identity=add_identity,
                channel_attention=channel_attention,
                eps=eps,
                momentum=momentum,
            ))
            self.add_module(f"stage{i + 1}", nn.Sequential(*stage))
            self.layers.append(f"stage{i + 1}")

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
    return ckpt


def load_rtmdet_pretrained_backbone(backbone, ckpt_path):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = extract_state_dict(ckpt)

    def remap_key(k):
        if not k.startswith("backbone."):
            return None
        k = k[len("backbone."):]
        k = k.replace(".conv2.depthwise_conv.", ".conv2.0.")
        k = k.replace(".conv2.pointwise_conv.", ".conv2.1.")
        return k

    filtered, remapped = {}, 0
    for k, v in state_dict.items():
        new_k = remap_key(k)
        if new_k is None:
            continue
        if new_k != k[len("backbone."):]:
            remapped += 1
        filtered[new_k] = v

    missing, unexpected = backbone.load_state_dict(filtered, strict=False)
    loaded_keys = set(filtered.keys()) - set(unexpected)
    summary = {
        "checkpoint": str(ckpt_path),
        "raw_keys": len(state_dict),
        "backbone_candidate_keys": len(filtered),
        "remapped_keys": remapped,
        "loaded_keys": len(loaded_keys),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }
    print(json.dumps(summary, indent=2))
    if len(missing):
        print("sample missing:", missing[:20])
    if len(unexpected):
        print("sample unexpected:", unexpected[:20])
    return summary


class _RTMDetMultiScaleBackbone(nn.Module):
    def __init__(self, pretrained_backbone_path, arch="P5",
                 deepen_factor=1.0, widen_factor=1.0, expand_ratio=0.5,
                 channel_attention=True, out_indices=(2, 3, 4),
                 fpn_dim=128, groups=8, eps=1e-3, momentum=0.01):
        super().__init__()
        self.fpn_dim = fpn_dim
        self.backbone = CSPNeXtBackboneFromRTMDet(
            arch=arch, deepen_factor=deepen_factor, widen_factor=widen_factor,
            expand_ratio=expand_ratio, channel_attention=channel_attention,
            out_indices=out_indices, eps=eps, momentum=momentum,
        )
        self.backbone_load_summary = load_rtmdet_pretrained_backbone(
            self.backbone, Path(pretrained_backbone_path))
        self.laterals = nn.ModuleList([
            nn.Conv2d(256, fpn_dim, 1),
            nn.Conv2d(512, fpn_dim, 1),
            nn.Conv2d(1024, fpn_dim, 1),
        ])
        self.smooth16 = ConvGNAct(fpn_dim, fpn_dim, k=3, s=1, p=1, groups=groups, act=True)
        self.smooth8 = ConvGNAct(fpn_dim, fpn_dim, k=3, s=1, p=1, groups=groups, act=True)
        self.out_proj = nn.Sequential(
            ConvGNAct(fpn_dim * 3, fpn_dim, k=1, s=1, p=0, groups=groups, act=True),
            ConvGNAct(fpn_dim, fpn_dim, k=3, s=1, p=1, groups=groups, act=True),
        )

    def freeze_all_stages(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_stages(self, n_last_stages=2):
        self.freeze_all_stages()
        stage_names = ["stage1", "stage2", "stage3", "stage4"]
        for name in stage_names[-n_last_stages:]:
            for p in getattr(self.backbone, name).parameters():
                p.requires_grad = True

    def forward(self, x):
        feat_s8, feat_s16, feat_s32 = self.backbone(x)
        lat8 = self.laterals[0](feat_s8)
        lat16 = self.laterals[1](feat_s16)
        p32 = self.laterals[2](feat_s32)
        p16 = self.smooth16(lat16 + F.interpolate(p32, size=lat16.shape[-2:], mode="bilinear", align_corners=False))
        p8 = self.smooth8(lat8 + F.interpolate(p16, size=lat8.shape[-2:], mode="bilinear", align_corners=False))
        p16_up = F.interpolate(p16, size=p8.shape[-2:], mode="bilinear", align_corners=False)
        p32_up = F.interpolate(p32, size=p8.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.out_proj(torch.cat([p8, p16_up, p32_up], dim=1))
        return {
            "feat_s8": feat_s8, "feat_s16": feat_s16, "feat_s32": feat_s32,
            "p8": p8, "p16": p16, "p32": p32, "fused": fused,
        }
