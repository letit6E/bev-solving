"""v7 model: RTMDet/CSPNeXt backbone + LSS view transform + StrongBEVEncoderDecoder.

The "Clean" suffix in the class name is preserved for checkpoint compatibility.
"""
import torch
import torch.nn as nn

from src.geometry import BEV_H, BEV_RES, BEV_W, X_RANGE, Y_RANGE
from src.models.cspnext import _RTMDetMultiScaleBackbone
from src.models.lss import LSSViewTransform2D
from src.models.strong_v4 import StrongBEVEncoderDecoder


class MultiCamBEVv7RTMDetCSPNeXtLSSClean(nn.Module):
    def __init__(self, num_rover_classes, rover_emb_dim=8, rover_cond_dim=8,
                 n_cameras=4, freeze_backbone=False,
                 pretrained_backbone_path="./rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth",
                 csp_arch="P5", csp_deepen_factor=1.0, csp_widen_factor=1.0,
                 csp_expand_ratio=0.5, csp_channel_attention=True,
                 csp_out_indices=(2, 3, 4),
                 fpn_dim=128, context_dim=80,
                 depth_bins=24, depth_min=1.0, depth_max=80.0,
                 world_z_min=-2.0, world_z_max=4.5,
                 bev_base_channels=96, bev_gn_groups=8):
        super().__init__()
        self.n_cameras = n_cameras
        self.rover_cond_dim = rover_cond_dim

        self.backbone = _RTMDetMultiScaleBackbone(
            pretrained_backbone_path=pretrained_backbone_path,
            arch=csp_arch, deepen_factor=csp_deepen_factor,
            widen_factor=csp_widen_factor, expand_ratio=csp_expand_ratio,
            channel_attention=csp_channel_attention,
            out_indices=csp_out_indices, fpn_dim=fpn_dim, groups=bev_gn_groups,
        )
        if freeze_backbone:
            self.backbone.freeze_all_stages()

        self.view_transform = LSSViewTransform2D(
            in_c=fpn_dim, context_c=context_dim,
            depth_bins=depth_bins, depth_min=depth_min, depth_max=depth_max,
            bev_h=BEV_H, bev_w=BEV_W, bev_res=BEV_RES,
            x_range=X_RANGE, y_range=Y_RANGE,
            z_min=world_z_min, z_max=world_z_max,
            groups=bev_gn_groups,
        )

        self.rover_embed = nn.Embedding(num_rover_classes, rover_emb_dim)
        nn.init.normal_(self.rover_embed.weight, std=0.02)
        self.rover_mlp = nn.Sequential(
            nn.Linear(rover_emb_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, rover_cond_dim),
            nn.ReLU(inplace=True),
        )

        self.bev_decoder = StrongBEVEncoderDecoder(
            in_c=context_dim + rover_cond_dim,
            base_c=bev_base_channels,
            groups=bev_gn_groups,
        )

    def forward_debug(self, images, intrinsics, car2cams, rover_ids):
        B, N, C, H, W = images.shape
        assert N == self.n_cameras
        x = images.reshape(B * N, C, H, W)
        back = self.backbone(x)
        feat_s8 = back["feat_s8"].reshape(B, N, *back["feat_s8"].shape[1:])
        feat_s16 = back["feat_s16"].reshape(B, N, *back["feat_s16"].shape[1:])
        feat_s32 = back["feat_s32"].reshape(B, N, *back["feat_s32"].shape[1:])
        fused = back["fused"].reshape(B, N, *back["fused"].shape[1:])
        bev, vt_debug = self.view_transform(fused, intrinsics, car2cams, image_hw=(H, W))
        rover_feat = self.rover_mlp(self.rover_embed(rover_ids)).view(B, self.rover_cond_dim, 1, 1)
        rover_map = rover_feat.expand(-1, -1, BEV_H, BEV_W)
        logits = self.bev_decoder(torch.cat([bev, rover_map], dim=1))
        return {
            "feat_s8": feat_s8, "feat_s16": feat_s16, "feat_s32": feat_s32,
            "image_fused": fused,
            "depth_logits": vt_debug["depth_logits"],
            "depth_prob": vt_debug["depth_prob"],
            "bev_raw": vt_debug["bev"],
            "valid_ratio": vt_debug["valid_ratio"],
            "logits": logits,
        }

    def forward(self, images, intrinsics, car2cams, rover_ids):
        dbg = self.forward_debug(images, intrinsics, car2cams, rover_ids)
        return torch.nan_to_num(dbg["logits"], nan=0.0, posinf=0.0, neginf=0.0)
