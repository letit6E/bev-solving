"""Parameter-free voxel projection shared by v1..v5.

For each BEV cell on a few height planes, project into every camera, sample
features with grid_sample, mean over visible cameras. This is the Simple-BEV
recipe (Harley et al., 2206.07959) — no learned lifting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.geometry import BEV_H, BEV_W, BEV_RES, X_RANGE, Y_RANGE, Z_LEVELS


def make_ego_voxels(bev_h=BEV_H, bev_w=BEV_W,
                    x_range=X_RANGE, y_range=Y_RANGE, z_levels=Z_LEVELS):
    xs = torch.linspace(x_range[0] + BEV_RES / 2, x_range[1] - BEV_RES / 2, bev_h)
    ys = torch.linspace(y_range[0] + BEV_RES / 2, y_range[1] - BEV_RES / 2, bev_w)
    zs = torch.tensor(z_levels, dtype=torch.float32)
    Z, X, Y = torch.meshgrid(zs, xs, ys, indexing="ij")
    ones = torch.ones_like(X)
    return torch.stack([X, Y, Z, ones], dim=-1)  # (Z, H, W, 4)


def project_and_sample(feat, ego_voxels, intrinsics, car2cams, img_hw):
    """feat: (B,N,C,Hf,Wf) -> aggregated: (B, C, Z, H, W)."""
    B, N, C, Hf, Wf = feat.shape
    Z, H, W, _ = ego_voxels.shape
    V = Z * H * W
    Hi, Wi = img_hw

    voxels = ego_voxels.reshape(-1, 4).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
    p_cam = torch.einsum("bnij,bnvj->bniv", car2cams, voxels)[:, :, :3]
    uv = torch.einsum("bnij,bnjv->bniv", intrinsics, p_cam)
    z = uv[:, :, 2]
    u_n = 2.0 * (uv[:, :, 0] / z.clamp(min=1e-3)) / Wi - 1.0
    v_n = 2.0 * (uv[:, :, 1] / z.clamp(min=1e-3)) / Hi - 1.0
    valid = (z > 0.1) & (u_n.abs() <= 1.0) & (v_n.abs() <= 1.0)

    grid = torch.stack([u_n, v_n], dim=-1).reshape(B * N, V, 1, 2)
    sampled = F.grid_sample(
        feat.reshape(B * N, C, Hf, Wf), grid,
        mode="bilinear", padding_mode="zeros", align_corners=False)
    sampled = sampled.squeeze(-1).reshape(B, N, C, V)

    vf = valid.float().unsqueeze(2)
    agg = (sampled * vf).sum(1) / vf.sum(1).clamp(min=1.0)
    return agg.reshape(B, C, Z, H, W)


class VoxelBEVHead(nn.Module):
    """Backbone-agnostic head. Subclasses just plug in `self.backbone` and `feat_dim`."""

    def __init__(self, feat_dim=128, proj_dim=64, decoder_cls=None):
        super().__init__()
        from src.models.decoder import SmallUNet
        self.feat_proj = nn.Conv2d(feat_dim, proj_dim, 1)
        self.register_buffer("ego_voxels", make_ego_voxels(), persistent=False)
        in_c = proj_dim * len(Z_LEVELS)
        self.bev_decoder = (decoder_cls or SmallUNet)(in_c=in_c, base_c=32, out_c=1)
        self.proj_dim = proj_dim

    def encode(self, images):
        B, N, C, H, W = images.shape
        feat = self.feat_proj(self.backbone(images.reshape(B * N, C, H, W)))
        Hf, Wf = feat.shape[-2:]
        return feat.reshape(B, N, self.proj_dim, Hf, Wf)

    def forward(self, images, intrinsics, car2cams):
        feat = self.encode(images)
        agg = project_and_sample(feat, self.ego_voxels, intrinsics, car2cams,
                                 images.shape[-2:])
        B, C, Z, H, W = agg.shape
        return self.bev_decoder(agg.reshape(B, C * Z, H, W))
