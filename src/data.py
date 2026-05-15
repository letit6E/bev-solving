"""BEV multi-camera dataset (4 cams -> tensors + intrinsics + extrinsics + GT)."""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import ImageFilter

from src.geometry import (
    CAMERA_NAMES, INTRINSICS_NAMES, CAR2CAM_NAMES, GT_NAME,
    IMAGENET_MEAN, IMAGENET_STD, resolve_info_path,
)
from src.splits import encode_rover

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _resize_intrinsic(K, src_hw, tgt_hw):
    sH, sW = src_hw
    tH, tW = tgt_hw
    K = K.copy().astype(np.float32)
    K[0, 0] *= tW / sW
    K[0, 2] *= tW / sW
    K[1, 1] *= tH / sH
    K[1, 2] *= tH / sH
    return K


class BEVDataset(Dataset):
    """Returns dict with images (4,3,H,W), intrinsics (4,3,3), car2cams (4,4,4), gt (1,188,126)."""

    def __init__(self, data_dir, mode="train", img_hw=(384, 768)):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.img_hw = img_hw
        self.info = pd.read_csv(self.data_dir / "info.csv", index_col=0)
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.info)

    def _resolve(self, p):
        return resolve_info_path(self.data_dir, p)

    def _load_camera(self, img_path, intr_path, car2cam_path):
        img = Image.open(self._resolve(img_path)).convert("RGB")
        src_hw = (img.size[1], img.size[0])
        img = img.resize((self.img_hw[1], self.img_hw[0]), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_t = self.normalize(img_t)
        K = np.load(self._resolve(intr_path))[:, :3]
        K = _resize_intrinsic(K, src_hw, self.img_hw)
        car2cam = np.load(self._resolve(car2cam_path)).astype(np.float32)
        return img_t, K, car2cam

    def _load_sample(self, idx):
        row = self.info.iloc[idx]
        imgs, Ks, Ms = [], [], []
        for cn, inn, cc in zip(CAMERA_NAMES, INTRINSICS_NAMES, CAR2CAM_NAMES):
            img_t, K, m = self._load_camera(row[cn], row[inn], row[cc])
            imgs.append(img_t)
            Ks.append(torch.from_numpy(K))
            Ms.append(torch.from_numpy(m))
        out = {
            "images": torch.stack(imgs, 0),
            "intrinsics": torch.stack(Ks, 0),
            "car2cams": torch.stack(Ms, 0),
            "info_idx": idx,
        }
        if self.mode != "test":
            gt = np.load(self._resolve(row[GT_NAME])).squeeze()
            gt = np.where(gt < 0, 255, gt).astype(np.int64)
            out["gt"] = torch.from_numpy(gt).unsqueeze(0)
        return out

    def __getitem__(self, idx):
        # Some PNGs in the dataset are truncated. Try a few neighbour indices on failure.
        last_err = None
        for k in range(5):
            try:
                return self._load_sample((idx + k) % len(self.info))
            except (OSError, ValueError) as e:
                last_err = e
        if self.mode == "test":
            H, W = self.img_hw
            return {
                "images": torch.zeros(4, 3, H, W),
                "intrinsics": torch.eye(3).unsqueeze(0).repeat(4, 1, 1),
                "car2cams": torch.eye(4).unsqueeze(0).repeat(4, 1, 1),
                "info_idx": idx,
            }
        raise RuntimeError(f"could not load idx={idx}: {last_err}")


class BEVDatasetAug(BEVDataset):
    """Adds per-camera random scale + crop with proper intrinsic update (Simple-BEV recipe)."""

    def __init__(self, data_dir, mode="train", img_hw=(448, 800),
                 aug=False, scale_range=(1.0, 1.2)):
        super().__init__(data_dir, mode, img_hw)
        self.aug = aug and mode == "train"
        self.scale_range = scale_range

    def _load_camera_aug(self, img_path, intr_path, car2cam_path, scale, dy, dx):
        img = Image.open(self._resolve(img_path)).convert("RGB")
        src_H, src_W = img.size[1], img.size[0]
        H_t, W_t = self.img_hw
        new_H, new_W = int(round(H_t * scale)), int(round(W_t * scale))
        img = img.resize((new_W, new_H), Image.BILINEAR).crop((dx, dy, dx + W_t, dy + H_t))
        arr = np.array(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        img_t = self.normalize(torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0)

        K = np.load(self._resolve(intr_path))[:, :3].copy().astype(np.float32)
        K[0, 0] *= new_W / src_W
        K[0, 2] *= new_W / src_W
        K[1, 1] *= new_H / src_H
        K[1, 2] *= new_H / src_H
        K[0, 2] -= dx
        K[1, 2] -= dy

        car2cam = np.load(self._resolve(car2cam_path)).astype(np.float32)
        return img_t, K, car2cam

    def _load_sample(self, idx):
        row = self.info.iloc[idx]
        H_t, W_t = self.img_hw
        imgs, Ks, Ms = [], [], []
        for cn, inn, cc in zip(CAMERA_NAMES, INTRINSICS_NAMES, CAR2CAM_NAMES):
            if self.aug:
                s = random.uniform(*self.scale_range)
                new_H, new_W = int(round(H_t * s)), int(round(W_t * s))
                dy = random.randint(0, max(0, new_H - H_t))
                dx = random.randint(0, max(0, new_W - W_t))
            else:
                s, dy, dx = 1.0, 0, 0
            img_t, K, m = self._load_camera_aug(row[cn], row[inn], row[cc], s, dy, dx)
            imgs.append(img_t)
            Ks.append(torch.from_numpy(K))
            Ms.append(torch.from_numpy(m))
        out = {
            "images": torch.stack(imgs, 0),
            "intrinsics": torch.stack(Ks, 0),
            "car2cams": torch.stack(Ms, 0),
            "info_idx": idx,
        }
        if self.mode != "test":
            gt = np.load(self._resolve(row[GT_NAME])).squeeze()
            gt = np.where(gt < 0, 255, gt).astype(np.int64)
            out["gt"] = torch.from_numpy(gt).unsqueeze(0)
        return out


def compute_coverage_weights(info_csv, cache_path=None, alpha=0.5, min_weight=0.1):
    """Coverage-aware WeightedRandomSampler weights. Penalises samples with mostly-ignored GT."""
    info_csv = Path(info_csv)
    if cache_path is not None and Path(cache_path).exists():
        return np.load(cache_path)

    info = pd.read_csv(info_csv, index_col=0)
    base = info_csv.parent
    cov = np.zeros(len(info), dtype=np.float32)
    for i, (_, row) in enumerate(info.iterrows()):
        gt = np.load(resolve_info_path(base, row[GT_NAME])).squeeze()
        cov[i] = (gt != 255).mean()
    weights = cov ** alpha + min_weight
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, weights)
    return weights


class BEVDatasetV4Clean(Dataset):
    """Letterbox dataset over an already-cleaned merged info DataFrame.

    Used from v6 onwards. Expects each row to carry a `__data_root` column added
    by `load_info_with_root` so paths resolve regardless of kaggle-safe rename.
    """

    def __init__(self, info_df, mode="train", img_hw=(384, 704),
                 aug=False, rover_vocab=None):
        self.info = info_df.reset_index(drop=True).copy()
        self.mode = mode
        self.img_hw = img_hw
        self.aug = aug and mode == "train"
        self.rover_vocab = rover_vocab or {"__other__": 0}
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.10, contrast=0.10, saturation=0.06, hue=0.02)

    def __len__(self):
        return len(self.info)

    def _resolve_path(self, row, key):
        return resolve_info_path(Path(row["__data_root"]), row[key])

    def _load_camera(self, row, cam_key, intr_key, c2c_key, scale_aug=1.0):
        img = Image.open(self._resolve_path(row, cam_key)).convert("RGB")
        intr_path = self._resolve_path(row, intr_key)
        car2cam_path = self._resolve_path(row, c2c_key)
        src_W, src_H = img.size
        H_t, W_t = self.img_hw

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

        if self.aug:
            if random.random() < 0.70:
                canvas = self.color_jitter(canvas)
            if random.random() < 0.15:
                canvas = canvas.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.8)))

        arr = np.array(canvas)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        img_t = self.normalize(torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0)

        K = np.load(intr_path)[:, :3].copy().astype(np.float32)
        K[0, 0] *= s; K[0, 2] *= s
        K[1, 1] *= s; K[1, 2] *= s
        K[0, 2] += pad_x
        K[1, 2] += pad_y
        K[0, 0] *= extra_s; K[0, 2] *= extra_s
        K[1, 1] *= extra_s; K[1, 2] *= extra_s
        K[0, 2] -= extra_dx
        K[1, 2] -= extra_dy

        car2cam = np.load(car2cam_path).astype(np.float32)
        return img_t, K, car2cam

    def _load_sample(self, idx):
        row = self.info.iloc[idx]
        scale_aug = random.uniform(1.0, 1.15) if self.aug else 1.0
        imgs, Ks, Ms = [], [], []
        for cn, inn, cc in zip(CAMERA_NAMES, INTRINSICS_NAMES, CAR2CAM_NAMES):
            img_t, K, m = self._load_camera(row, cn, inn, cc, scale_aug)
            imgs.append(img_t)
            Ks.append(torch.from_numpy(K))
            Ms.append(torch.from_numpy(m))
        out = {
            "images": torch.stack(imgs, 0),
            "intrinsics": torch.stack(Ks, 0),
            "car2cams": torch.stack(Ms, 0),
            "rover_id": torch.tensor(
                encode_rover(row.get("rover", "__other__"), self.rover_vocab),
                dtype=torch.long),
            "info_idx": idx,
        }
        if self.mode != "test":
            gt = np.load(self._resolve_path(row, GT_NAME)).squeeze()
            gt = np.where(gt < 0, 255, gt).astype(np.int64)
            out["gt"] = torch.from_numpy(gt).unsqueeze(0)
        return out

    def __getitem__(self, idx):
        last_err = None
        for k in range(5):
            try:
                return self._load_sample((idx + k) % len(self.info))
            except (OSError, ValueError, FileNotFoundError) as e:
                last_err = e
        raise RuntimeError(f"could not load idx={idx}: {last_err}")
