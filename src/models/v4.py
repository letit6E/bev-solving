"""v4: ResNet-50 backbone, letterbox resize, rover embedding broadcast on BEV.

Letterbox keeps aspect ratio — fixes rover `nack` (768×959) projection issues.
Rover embedding adds a per-vehicle bias on the BEV feature before the decoder.
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image

from src.data import BEVDataset
from src.geometry import (
    CAMERA_NAMES, CAR2CAM_NAMES, INTRINSICS_NAMES, GT_NAME,
)
from src.models.decoder import SmallUNet
from src.models.voxel import make_ego_voxels, project_and_sample


def build_rover_vocab(*info_csvs):
    rovers = set()
    for p in info_csvs:
        df = pd.read_csv(p, index_col=0)
        if "rover" in df.columns:
            rovers.update(df["rover"].unique())
    return {r: i for i, r in enumerate(sorted(rovers))}


class BEVDatasetV4(BEVDataset):
    def __init__(self, data_dir, mode="train", img_hw=(384, 704),
                 aug=False, rover_vocab=None):
        super().__init__(data_dir, mode, img_hw)
        self.aug = aug and mode == "train"
        self.rover_vocab = rover_vocab or {}
        self._unknown = len(self.rover_vocab)

    def _load_camera(self, img_path, intr_path, car2cam_path, scale_aug=1.0):
        img = Image.open(self._resolve(img_path)).convert("RGB")
        src_W, src_H = img.size
        H_t, W_t = self.img_hw

        # Letterbox: uniform scale + pad to target.
        s = min(W_t / src_W, H_t / src_H)
        new_W, new_H = int(round(src_W * s)), int(round(src_H * s))
        canvas = Image.new("RGB", (W_t, H_t), 0)
        pad_x = (W_t - new_W) // 2
        pad_y = (H_t - new_H) // 2
        canvas.paste(img.resize((new_W, new_H), Image.BILINEAR), (pad_x, pad_y))

        extra_s, extra_dx, extra_dy = 1.0, 0, 0
        if scale_aug > 1.0:
            sH, sW = int(round(H_t * scale_aug)), int(round(W_t * scale_aug))
            canvas = canvas.resize((sW, sH), Image.BILINEAR)
            extra_dx = random.randint(0, sW - W_t)
            extra_dy = random.randint(0, sH - H_t)
            canvas = canvas.crop((extra_dx, extra_dy, extra_dx + W_t, extra_dy + H_t))
            extra_s = scale_aug

        arr = np.array(canvas)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        img_t = self.normalize(torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0)

        K = np.load(self._resolve(intr_path))[:, :3].copy().astype(np.float32)
        K[0, 0] *= s; K[0, 2] *= s
        K[1, 1] *= s; K[1, 2] *= s
        K[0, 2] += pad_x
        K[1, 2] += pad_y
        K[0, 0] *= extra_s; K[0, 2] *= extra_s
        K[1, 1] *= extra_s; K[1, 2] *= extra_s
        K[0, 2] -= extra_dx
        K[1, 2] -= extra_dy

        car2cam = np.load(self._resolve(car2cam_path)).astype(np.float32)
        return img_t, K, car2cam

    def _load_sample(self, idx):
        row = self.info.iloc[idx]
        scale_aug = random.uniform(1.0, 1.15) if self.aug else 1.0
        imgs, Ks, Ms = [], [], []
        for cn, inn, cc in zip(CAMERA_NAMES, INTRINSICS_NAMES, CAR2CAM_NAMES):
            img_t, K, m = self._load_camera(row[cn], row[inn], row[cc], scale_aug)
            imgs.append(img_t)
            Ks.append(torch.from_numpy(K))
            Ms.append(torch.from_numpy(m))
        out = {
            "images": torch.stack(imgs, 0),
            "intrinsics": torch.stack(Ks, 0),
            "car2cams": torch.stack(Ms, 0),
            "rover_id": torch.tensor(
                self.rover_vocab.get(row.get("rover", "?"), self._unknown), dtype=torch.long),
            "info_idx": idx,
        }
        if self.mode != "test":
            gt = np.load(self._resolve(row[GT_NAME])).squeeze()
            gt = np.where(gt < 0, 255, gt).astype(np.int64)
            out["gt"] = torch.from_numpy(gt).unsqueeze(0)
        return out


class _R50Stem(nn.Module):
    def __init__(self):
        super().__init__()
        rn = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool)
        self.layer1, self.layer2 = rn.layer1, rn.layer2
        self.proj = nn.Conv2d(512, 128, 1)

    def forward(self, x):
        return self.proj(self.layer2(self.layer1(self.stem(x))))


class MultiCamBEVv4(nn.Module):
    def __init__(self, num_rovers=64, rover_emb_dim=32, n_cameras=4):
        super().__init__()
        self.n_cameras = n_cameras
        self.backbone = _R50Stem()
        self.feat_proj = nn.Conv2d(128, 64, 1)
        self.register_buffer("ego_voxels", make_ego_voxels(), persistent=False)
        self.rover_emb_dim = rover_emb_dim
        self.rover_embed = nn.Embedding(num_rovers + 1, rover_emb_dim)
        nn.init.normal_(self.rover_embed.weight, std=0.02)
        z = self.ego_voxels.shape[0]
        self.bev_decoder = SmallUNet(in_c=64 * z + rover_emb_dim, base_c=32, out_c=1)

    def forward(self, images, intrinsics, car2cams, rover_ids=None):
        B, N, C, Hi, Wi = images.shape
        feat = self.feat_proj(self.backbone(images.reshape(B * N, C, Hi, Wi)))
        Hf, Wf = feat.shape[-2:]
        feat = feat.reshape(B, N, 64, Hf, Wf)
        agg = project_and_sample(feat, self.ego_voxels, intrinsics, car2cams, (Hi, Wi))
        B_, _, Z, H, W = agg.shape
        agg = agg.reshape(B, 64 * Z, H, W)

        if rover_ids is None:
            rover_ids = torch.zeros(B, dtype=torch.long, device=agg.device)
        emb = self.rover_embed(rover_ids).view(B, self.rover_emb_dim, 1, 1).expand(-1, -1, H, W)
        return self.bev_decoder(torch.cat([agg, emb], 1))
