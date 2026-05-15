"""LSS view transform and DINOv2 multi-scale backbone used by v6/v7 notebooks.

This is the canonical post-image path: DINOv2 ViT-B/14 features fused across 4
taps, then Lift-Splat-Shoot with a learned per-pixel depth distribution. See
notebooks/stage_v6_dinov2_lss/ for the wiring around it.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def gn_groups(channels: int, requested: int = 8) -> int:
    g = min(requested, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return max(g, 1)


class ConvGNAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, groups=8, act=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False),
            nn.GroupNorm(gn_groups(out_c, groups), out_c),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock2d(nn.Module):
    def __init__(self, in_c, out_c, stride=1, groups=8):
        super().__init__()
        self.conv1 = ConvGNAct(in_c, out_c, k=3, s=stride, p=1, groups=groups, act=True)
        self.conv2 = ConvGNAct(out_c, out_c, k=3, s=1, p=1, groups=groups, act=False)
        if stride != 1 or in_c != out_c:
            self.skip = ConvGNAct(in_c, out_c, k=1, s=stride, p=0, groups=groups, act=False)
        else:
            self.skip = nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class ASPP2d(nn.Module):
    def __init__(self, in_c, out_c, rates=(1, 3, 6), groups=8):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=r, dilation=r, bias=False),
                nn.GroupNorm(gn_groups(out_c, groups), out_c),
                nn.SiLU(inplace=True),
            )
            for r in rates
        ])
        self.proj = ConvGNAct(out_c * len(rates), out_c, k=1, s=1, p=0, groups=groups, act=True)

    def forward(self, x):
        xs = [b(x) for b in self.branches]
        return self.proj(torch.cat(xs, dim=1))


class _DINOv2MultiScaleBackbone(nn.Module):
    def __init__(self,
                 hub_repo: str = 'facebookresearch/dinov2',
                 backbone_name: str = 'dinov2_vitb14',
                 out_dim: int = 768,
                 patch_size: int = 14,
                 tap_layers=(2, 5, 8, 11),
                 neck_dim: int = 128,
                 groups: int = 8):
        super().__init__()
        self.hub_repo = hub_repo
        self.backbone_name = backbone_name
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.tap_layers = tuple(tap_layers)
        self.neck_dim = neck_dim

        self.vit = self._load_hub_model()
        self.laterals = nn.ModuleList([nn.Conv2d(out_dim, neck_dim, 1) for _ in self.tap_layers])
        self.fuse = nn.Sequential(
            ConvGNAct(len(self.tap_layers) * neck_dim, neck_dim, k=3, s=1, p=1, groups=groups, act=True),
            ConvGNAct(neck_dim, neck_dim, k=3, s=1, p=1, groups=groups, act=True),
        )
        self.down1 = ConvGNAct(neck_dim, neck_dim, k=3, s=2, p=1, groups=groups, act=True)
        self.down2 = ConvGNAct(neck_dim, neck_dim, k=3, s=2, p=1, groups=groups, act=True)
        self.neck_out = ConvGNAct(neck_dim * 3, neck_dim, k=1, s=1, p=0, groups=groups, act=True)

    def _load_hub_model(self):
        last_err = None
        attempts = [
            dict(repo_or_dir=self.hub_repo, model=self.backbone_name),
            dict(repo_or_dir=self.hub_repo, model=self.backbone_name, pretrained=True),
            dict(repo_or_dir=self.hub_repo, model=self.backbone_name, source='github'),
            dict(repo_or_dir=self.hub_repo, model=self.backbone_name, source='github', pretrained=True),
        ]
        for kwargs in attempts:
            try:
                return torch.hub.load(**kwargs)
            except Exception as e:
                last_err = e
        raise RuntimeError(
            f'Failed to load DINOv2 backbone {self.backbone_name} from {self.hub_repo}. '
            f'Last error: {last_err}'
        )

    def _reshape_tokens(self, tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B = tokens.shape[0]
        expected_tokens = (H // self.patch_size) * (W // self.patch_size)
        if tokens.ndim != 3:
            raise RuntimeError(f'Unexpected token shape: {tuple(tokens.shape)}')
        if tokens.shape[1] == expected_tokens + 1:
            tokens = tokens[:, 1:, :]
        elif tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                f'Unexpected number of DINO tokens: got {tokens.shape[1]}, expected {expected_tokens} '
                f'for img_hw={(H, W)}'
            )
        Hp = H // self.patch_size
        Wp = W // self.patch_size
        return tokens.transpose(1, 2).reshape(B, self.out_dim, Hp, Wp).contiguous()

    def _extract_intermediate(self, x: torch.Tensor):
        H, W = x.shape[-2:]
        try:
            feats = self.vit.get_intermediate_layers(
                x,
                n=list(self.tap_layers),
                reshape=True,
                return_class_token=False,
            )
        except Exception:
            feats = self.vit.get_intermediate_layers(
                x,
                n=len(self.tap_layers),
                reshape=True,
                return_class_token=False,
            )

        out = []
        for feat in feats:
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            if feat.ndim == 3:
                feat = self._reshape_tokens(feat, H, W)
            out.append(feat)
        if len(out) != len(self.tap_layers):
            raise RuntimeError(f'Expected {len(self.tap_layers)} intermediate features, got {len(out)}')
        return out

    def forward(self, x):
        feats = self._extract_intermediate(x)
        laterals = [proj(feat) for proj, feat in zip(self.laterals, feats)]
        p0 = self.fuse(torch.cat(laterals, dim=1))
        p1 = self.down1(p0)
        p2 = self.down2(p1)
        p1_up = F.interpolate(p1, size=p0.shape[-2:], mode='bilinear', align_corners=False)
        p2_up = F.interpolate(p2, size=p0.shape[-2:], mode='bilinear', align_corners=False)
        fused = self.neck_out(torch.cat([p0, p1_up, p2_up], dim=1))
        return {
            'tap_features': feats,
            'p0': p0,
            'p1': p1,
            'p2': p2,
            'fused': fused,
        }


class LSSViewTransform2D(nn.Module):
    def __init__(self,
                 in_c: int,
                 context_c: int,
                 depth_bins: int,
                 depth_min: float,
                 depth_max: float,
                 bev_h: int,
                 bev_w: int,
                 bev_res: float,
                 x_range,
                 y_range,
                 z_min: float,
                 z_max: float,
                 groups: int = 8,
                 depth_spacing: str = 'linear'):
        super().__init__()
        self.context_c = context_c
        self.depth_bins = depth_bins
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_res = float(bev_res)
        self.x_range = x_range
        self.y_range = y_range
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.depth_spacing = str(depth_spacing)
        if self.depth_spacing not in {'linear', 'log'}:
            raise ValueError(f'Unsupported depth_spacing: {self.depth_spacing}')
        if self.depth_spacing == 'log' and self.depth_min <= 0:
            raise ValueError('depth_min must be > 0 for log depth spacing')

        self.depth_head = nn.Sequential(
            ConvGNAct(in_c, in_c, k=3, s=1, p=1, groups=groups, act=True),
            nn.Conv2d(in_c, depth_bins, 1),
        )
        self.context_head = nn.Sequential(
            ConvGNAct(in_c, in_c, k=3, s=1, p=1, groups=groups, act=True),
            nn.Conv2d(in_c, context_c, 1),
        )

    def _build_depths(self, device, dtype):
        if self.depth_spacing == 'log':
            return torch.exp(torch.linspace(math.log(self.depth_min), math.log(self.depth_max), self.depth_bins, device=device, dtype=dtype))
        return torch.linspace(self.depth_min, self.depth_max, self.depth_bins, device=device, dtype=dtype)

    def _build_frustum(self, Hf: int, Wf: int, Hi: int, Wi: int, device, dtype):
        depths = self._build_depths(device, dtype)
        xs = (torch.arange(Wf, device=device, dtype=dtype) + 0.5) * (Wi / Wf)
        ys = (torch.arange(Hf, device=device, dtype=dtype) + 0.5) * (Hi / Hf)
        d, y, x = torch.meshgrid(depths, ys, xs, indexing='ij')
        return x, y, d

    def forward(self, feat_2d: torch.Tensor, intrinsics: torch.Tensor, car2cams: torch.Tensor, image_hw):
        B, N, C, Hf, Wf = feat_2d.shape
        Hi, Wi = image_hw

        feat_bn = feat_2d.reshape(B * N, C, Hf, Wf)
        depth_logits = self.depth_head(feat_bn).float()
        context = self.context_head(feat_bn).float()

        depth_prob = torch.softmax(depth_logits, dim=1)
        depth_prob = depth_prob.reshape(B, N, self.depth_bins, Hf, Wf)
        context = context.reshape(B, N, self.context_c, Hf, Wf)

        x_img, y_img, depth_vals = self._build_frustum(Hf, Wf, Hi, Wi, feat_2d.device, torch.float32)
        x_img = x_img.view(1, 1, self.depth_bins, Hf, Wf)
        y_img = y_img.view(1, 1, self.depth_bins, Hf, Wf)
        depth_vals = depth_vals.view(1, 1, self.depth_bins, Hf, Wf)

        intrinsics = intrinsics.float()
        car2cams = car2cams.float()
        cam2cars = torch.inverse(car2cams.reshape(B * N, 4, 4)).reshape(B, N, 4, 4)

        fx = intrinsics[..., 0, 0].view(B, N, 1, 1, 1)
        fy = intrinsics[..., 1, 1].view(B, N, 1, 1, 1)
        cx = intrinsics[..., 0, 2].view(B, N, 1, 1, 1)
        cy = intrinsics[..., 1, 2].view(B, N, 1, 1, 1)

        X = (x_img - cx) / fx * depth_vals
        Y = (y_img - cy) / fy * depth_vals
        Z = depth_vals.expand(B, N, -1, -1, -1)
        ones = torch.ones_like(Z)
        pts_cam = torch.stack([X, Y, Z, ones], dim=-1)
        pts_car = torch.einsum('bnij,bndhwj->bndhwi', cam2cars, pts_cam)

        world_x = pts_car[..., 0]
        world_y = pts_car[..., 1]
        world_z = pts_car[..., 2]

        x_idx = torch.floor((world_x - self.x_range[0]) / self.bev_res).long()
        y_idx = torch.floor((world_y - self.y_range[0]) / self.bev_res).long()
        valid = (
            (x_idx >= 0) & (x_idx < self.bev_h) &
            (y_idx >= 0) & (y_idx < self.bev_w) &
            (world_z >= self.z_min) & (world_z <= self.z_max)
        )
        linear_idx = x_idx * self.bev_w + y_idx

        feat_vol = context.unsqueeze(3) * depth_prob.unsqueeze(2)
        bev = feat_2d.new_zeros(B, self.context_c, self.bev_h * self.bev_w, dtype=torch.float32)
        counts = feat_2d.new_zeros(B, 1, self.bev_h * self.bev_w, dtype=torch.float32)

        for b in range(B):
            idx_b = linear_idx[b].reshape(-1)
            valid_b = valid[b].reshape(-1)
            if not valid_b.any():
                continue
            feat_b = feat_vol[b].permute(1, 0, 2, 3, 4).reshape(self.context_c, -1)
            idx_valid = idx_b[valid_b]
            feat_valid = feat_b[:, valid_b]
            bev[b].scatter_add_(1, idx_valid.unsqueeze(0).expand(self.context_c, -1), feat_valid)
            counts[b].scatter_add_(1, idx_valid.unsqueeze(0), torch.ones(1, idx_valid.numel(), device=feat_2d.device, dtype=torch.float32))

        bev = bev / counts.clamp(min=1.0)
        bev = bev.reshape(B, self.context_c, self.bev_h, self.bev_w)
        bev = torch.nan_to_num(bev, nan=0.0, posinf=0.0, neginf=0.0)

        debug = {
            'depth_logits': depth_logits.reshape(B, N, self.depth_bins, Hf, Wf),
            'depth_prob': depth_prob,
            'context': context,
            'bev': bev,
            'valid_ratio': valid.float().mean().item(),
            'depth_values': self._build_depths(feat_2d.device, torch.float32).detach().cpu(),
        }
        return bev, debug

