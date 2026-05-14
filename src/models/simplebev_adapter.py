"""Wrap Simple-BEV's Segnet (nuScenes-pretrained) so we can score it zero-shot.

Their grid is 200×200 centred on the ego, covering ±50m forward/lateral. Ours
is 188×126, 0..150m forward, ±50m lateral. We shift their `scene_centroid` to
(0, 1, 50) so their forward range becomes 0..100m — overlaps better with our
ROI. The remaining 100..150m simply stays zero in the resampled output.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path("external/simple_bev")


def _patch_segnet_source(repo=REPO):
    """Replace hardcoded `.cuda()` calls in nets/segnet.py with register_buffer."""
    seg = repo / "nets" / "segnet.py"
    text = seg.read_text()
    needles = [
        ("self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float().cuda()",
         "self.register_buffer('mean', torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float())"),
        ("self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float().cuda()",
         "self.register_buffer('std', torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float())"),
    ]
    changed = False
    for old, new in needles:
        if old in text:
            text = text.replace(old, new)
            changed = True
    if changed:
        seg.write_text(text)


def _setup_imports(repo=REPO):
    if not repo.exists():
        raise FileNotFoundError(f"clone Simple-BEV into {repo}")
    _patch_segnet_source(repo)
    p = str(repo.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _find_ckpt(repo=REPO):
    cands = list(repo.rglob("model-*.pth")) + list(repo.rglob("model-*.pt"))
    if not cands:
        raise FileNotFoundError(f"no checkpoint under {repo}")
    return max(cands, key=lambda p: p.stat().st_size)


class SimpleBEVAdapter(nn.Module):
    def __init__(self, ckpt_path=None, device="cuda",
                 Z=200, Y=8, X=200,
                 bounds=(-50, 50, -5, 5, -50, 50),
                 scene_centroid=(0.0, 1.0, 50.0)):
        super().__init__()
        _setup_imports()
        from nets.segnet import Segnet
        import utils.vox

        self.device = torch.device(device)
        self.Z, self.Y, self.X = Z, Y, X
        cx, cy, cz = scene_centroid
        self.bounds_eff = (
            bounds[0] + cx, bounds[1] + cx,
            bounds[2] + cy, bounds[3] + cy,
            bounds[4] + cz, bounds[5] + cz,
        )
        centroid_t = torch.tensor([scene_centroid], device=self.device, dtype=torch.float32)
        self.vox_util = utils.vox.Vox_util(
            Z, Y, X, scene_centroid=centroid_t, bounds=bounds, assert_cube=False)

        self.model = Segnet(
            Z=Z, Y=Y, X=X, vox_util=self.vox_util,
            encoder_type="res101", use_radar=False, use_lidar=False,
            do_rgbcompress=True, rand_flip=False)
        ckpt = torch.load(Path(ckpt_path) if ckpt_path else _find_ckpt(), map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt)
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        self.model.load_state_dict(sd, strict=False)
        self.model.to(self.device).eval()

    def _convert_inputs(self, images_imagenet, intrinsics, car2cams):
        B, N = images_imagenet.shape[:2]
        device = images_imagenet.device
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
        rgb = images_imagenet * std + mean - 0.5

        pix_T = torch.zeros(B, N, 4, 4, device=device, dtype=intrinsics.dtype)
        pix_T[..., :3, :3] = intrinsics
        pix_T[..., 3, 3] = 1.0

        # cam0_T_camXs[i] = car_to_cam[0] @ inv(car_to_cam[i])
        cam0_T_camXs = torch.matmul(car2cams[:, 0:1], torch.linalg.inv(car2cams))
        return rgb, pix_T, cam0_T_camXs

    @torch.no_grad()
    def forward(self, images_imagenet, intrinsics, car2cams,
                target_hw=(188, 126),
                target_x_range=(0.0, 150.4), target_y_range=(-50.4, 50.4),
                swap_lateral=False):
        rgb, pix_T, cam0 = self._convert_inputs(images_imagenet, intrinsics, car2cams)
        rgb, pix_T, cam0 = rgb.to(self.device), pix_T.to(self.device), cam0.to(self.device)
        _, _, seg_e, _, _ = self.model(
            rgb_camXs=rgb.float(), pix_T_cams=pix_T.float(),
            cam0_T_camXs=cam0.float(), vox_util=self.vox_util, rad_occ_mem0=None)

        H_t, W_t = target_hw
        device = seg_e.device
        xs = torch.linspace(target_x_range[0] + 0.4, target_x_range[1] - 0.4, H_t, device=device)
        ys = torch.linspace(target_y_range[0] + 0.4, target_y_range[1] - 0.4, W_t, device=device)
        if swap_lateral:
            ys = -ys
        zmin, zmax = self.bounds_eff[4], self.bounds_eff[5]
        xmin, xmax = self.bounds_eff[0], self.bounds_eff[1]
        z_norm = (xs - zmin) / (zmax - zmin) * 2 - 1
        x_norm = (ys - xmin) / (xmax - xmin) * 2 - 1
        zz, yy = torch.meshgrid(z_norm, x_norm, indexing="ij")
        grid = torch.stack([yy, zz], dim=-1).unsqueeze(0).expand(seg_e.shape[0], -1, -1, -1)
        return F.grid_sample(seg_e, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
