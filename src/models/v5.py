"""v5: calibration-aware FiLM + specialist branch for top test rovers.

The 6-dim per-camera rig features (translation + forward axis) and a 12-dim
global rig summary feed FiLM modulators that warp image features per camera
and the BEV feature per sample. A specialist embedding (top-12 test rovers)
gates a bias-only correction on the BEV side. Rationale: rovers differ by
camera mount more than they differ by neighbourhood, so we condition on the
mount geometry directly instead of just on a rover_id.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision

from src.geometry import GT_NAME
from src.models.decoder import SmallUNet
from src.models.v4 import BEVDatasetV4
from src.models.voxel import make_ego_voxels, project_and_sample


def _pose_features(car2cam):
    cam2car = np.linalg.inv(car2cam)
    return np.concatenate([cam2car[:3, 3], cam2car[:3, 2]]).astype(np.float32)


def build_rig_features(intrinsics, car2cams, img_hw):
    H_t, W_t = img_hw
    cam_feats, poses = [], []
    for K, M in zip(intrinsics, car2cams):
        pose = _pose_features(M)
        poses.append(pose)
        cam_feats.append(np.array([
            K[0, 0] / W_t, K[1, 1] / H_t, K[0, 2] / W_t, K[1, 2] / H_t,
            pose[0] / 10, pose[1] / 10, pose[2] / 10,
            pose[3], pose[4], pose[5],
        ], dtype=np.float32))
    cam_feats = np.stack(cam_feats)
    poses = np.stack(poses)
    left, right, mid, far = poses[2], poses[3], poses[0], poses[1]
    g = np.array([
        cam_feats[:, 0].mean(), cam_feats[:, 0].std(),
        cam_feats[:, 1].mean(), cam_feats[:, 1].std(),
        abs(left[1] - right[1]) / 10, abs(mid[0] - far[0]) / 10, abs(mid[2] - far[2]) / 10,
        poses[:, 0].mean() / 10, poses[:, 1].mean() / 10, poses[:, 2].mean() / 10,
        poses[:, 0].std() / 10, poses[:, 1].std() / 10,
    ], dtype=np.float32)
    return cam_feats, g


def build_specialist_vocab(train_csv, test_csv, min_train_count=40, topk=12):
    train = pd.read_csv(train_csv, index_col=0)
    test = pd.read_csv(test_csv, index_col=0)
    tcnt = Counter(train["rover"])
    selected = []
    for rover, _ in Counter(test["rover"]).most_common():
        if tcnt.get(rover, 0) >= min_train_count:
            selected.append(rover)
        if len(selected) >= topk:
            break
    return {r: i for i, r in enumerate(selected)}


class BEVDatasetV5(BEVDatasetV4):
    def __init__(self, data_dir, mode="train", img_hw=(384, 704),
                 aug=False, rover_vocab=None, specialist_vocab=None):
        super().__init__(data_dir, mode, img_hw, aug, rover_vocab)
        self.specialist_vocab = specialist_vocab or {}
        self._unknown_spec = len(self.specialist_vocab)

    def _load_sample(self, idx):
        out = super()._load_sample(idx)
        cam_rig, global_rig = build_rig_features(
            out["intrinsics"].numpy(), out["car2cams"].numpy(), self.img_hw)
        out["cam_rig"] = torch.from_numpy(cam_rig)
        out["global_rig"] = torch.from_numpy(global_rig)
        rover = self.info.iloc[idx].get("rover", "?")
        out["specialist_id"] = torch.tensor(
            self.specialist_vocab.get(rover, self._unknown_spec), dtype=torch.long)
        return out


class _R34Stem(nn.Module):
    def __init__(self):
        super().__init__()
        rn = torchvision.models.resnet34(
            weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool)
        self.layer1, self.layer2 = rn.layer1, rn.layer2
        self.proj = nn.Conv2d(128, 128, 1)

    def forward(self, x):
        return self.proj(self.layer2(self.layer1(self.stem(x))))


class _FiLM(nn.Module):
    def __init__(self, cond_dim, feat_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.ReLU(True), nn.Linear(hidden, feat_dim * 2))

    def forward(self, c):
        return self.net(c).chunk(2, dim=-1)


class MultiCamBEVv5(nn.Module):
    def __init__(self, num_rovers, num_specialists,
                 rover_emb_dim=24, spec_emb_dim=16, n_cameras=4):
        super().__init__()
        self.n_cameras = n_cameras
        self.backbone = _R34Stem()
        self.feat_proj = nn.Conv2d(128, 64, 1)
        self.cam_film = _FiLM(10, 64)
        self.rover_embed = nn.Embedding(num_rovers + 1, rover_emb_dim)
        self.spec_embed = nn.Embedding(num_specialists + 1, spec_emb_dim)
        nn.init.normal_(self.rover_embed.weight, std=0.02)
        nn.init.normal_(self.spec_embed.weight, std=0.02)
        self.register_buffer("ego_voxels", make_ego_voxels(), persistent=False)
        Z = self.ego_voxels.shape[0]
        bev_cond = 12 + rover_emb_dim + spec_emb_dim
        self.bev_film = _FiLM(bev_cond, 64 * Z, hidden=96)
        self.spec_gate = nn.Sequential(
            nn.Linear(bev_cond, 32), nn.ReLU(True), nn.Linear(32, 1), nn.Sigmoid())
        self.bev_decoder = SmallUNet(in_c=64 * Z, base_c=32, out_c=1)

    def forward(self, images, intrinsics, car2cams,
                rover_ids=None, cam_rig=None, global_rig=None, specialist_ids=None):
        B, N, C, Hi, Wi = images.shape
        feat = self.feat_proj(self.backbone(images.reshape(B * N, C, Hi, Wi)))
        Hf, Wf = feat.shape[-2:]
        feat = feat.reshape(B, N, 64, Hf, Wf)

        if cam_rig is not None:
            g, b = self.cam_film(cam_rig.reshape(B * N, -1))
            feat = feat * (1.0 + 0.1 * g.view(B, N, 64, 1, 1)) + 0.1 * b.view(B, N, 64, 1, 1)

        agg = project_and_sample(feat, self.ego_voxels, intrinsics, car2cams, (Hi, Wi))
        _, _, Z, H, W = agg.shape
        agg = agg.reshape(B, 64 * Z, H, W)

        if rover_ids is None:
            rover_ids = torch.zeros(B, dtype=torch.long, device=agg.device)
        if specialist_ids is None:
            specialist_ids = torch.zeros(B, dtype=torch.long, device=agg.device)
        if global_rig is None:
            global_rig = torch.zeros(B, 12, dtype=agg.dtype, device=agg.device)

        cond = torch.cat([global_rig, self.rover_embed(rover_ids),
                          self.spec_embed(specialist_ids)], -1)
        g, b = self.bev_film(cond)
        gate = self.spec_gate(cond).view(B, 1, 1, 1)
        agg = agg * (1.0 + 0.1 * g.view(B, -1, 1, 1)) + 0.1 * gate * b.view(B, -1, 1, 1)
        return self.bev_decoder(agg)


def dump_v5_metadata(out_path, rover_vocab, specialist_vocab, notes=None):
    Path(out_path).write_text(json.dumps({
        "rover_vocab_size": len(rover_vocab),
        "specialist_vocab_size": len(specialist_vocab),
        "specialist_rovers": specialist_vocab,
        "notes": notes or {},
    }, indent=2, ensure_ascii=False))
