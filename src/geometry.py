"""BEV grid conventions fixed once after sanity check.

ego frame: X forward, Y left, Z up.
car_to_cam: p_cam = car_to_cam @ p_ego_h.
intrinsic stored as (3, 4) = [K | 0].
GT: 0 free, 1 occupied, 255 ignore.
"""
from pathlib import Path

CAMERA_NAMES = [
    "/camera/inner/frontal/middle",
    "/camera/inner/frontal/far",
    "/side/left/forward",
    "/side/right/forward",
]
INTRINSICS_NAMES = [n + "/intrinsic_params" for n in CAMERA_NAMES]
CAR2CAM_NAMES = [n + "/car_to_cam" for n in CAMERA_NAMES]
GT_NAME = "gt_occupancy_grid"

BEV_H, BEV_W = 188, 126
BEV_RES = 0.8
X_RANGE = (0.0, BEV_H * BEV_RES)
Y_RANGE = (-BEV_W * BEV_RES / 2, BEV_W * BEV_RES / 2)
Z_LEVELS = (0.3, 1.0, 2.0, 3.0)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_info_path(base_dir, p):
    p = Path(p)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p
    cand = Path(base_dir) / p
    if cand.exists():
        return cand
    return Path(base_dir).parent / p
