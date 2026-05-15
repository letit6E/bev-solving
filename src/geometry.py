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


def kaggle_safe_name(name):
    return name.replace(":", "_")


def resolve_info_path(base_dir, p):
    """Find a dataset path on disk.

    Handles raw paths and kaggle-safe layouts where `:` was replaced with `_` and
    anchor dirs like `autonomy_yandex_dataset_*` may differ between layouts.
    """
    p = Path(str(p))
    base_dir = Path(base_dir)
    seen = set()

    def take(cand):
        cand = Path(cand)
        key = str(cand)
        if key in seen:
            return None
        seen.add(key)
        return cand if cand.exists() else None

    for cand in (p, base_dir / p, base_dir.parent / p):
        hit = take(cand)
        if hit:
            return hit

    parts = list(p.parts)
    anchors = [
        "autonomy_yandex_dataset_train", "autonomy_yandex_dataset_val",
        "autonomy_yandex_dataset_test",
        "autonomy_yandex_dataset_train_kaggle_safe",
        "autonomy_yandex_dataset_val_kaggle_safe",
        "autonomy_yandex_dataset_test_kaggle_safe",
    ]
    for anchor in anchors:
        if anchor in parts:
            i = parts.index(anchor)
            rel = Path(*parts[i + 1:])
            for cand in (base_dir / rel, base_dir.parent / rel):
                hit = take(cand)
                if hit:
                    return hit
            safe_rel = Path(*[kaggle_safe_name(x) for x in rel.parts])
            for cand in (base_dir / safe_rel, base_dir.parent / safe_rel):
                hit = take(cand)
                if hit:
                    return hit

    safe_p = Path(*[kaggle_safe_name(x) for x in parts])
    for cand in (base_dir / safe_p, base_dir.parent / safe_p):
        hit = take(cand)
        if hit:
            return hit

    # Last-ditch fallback so callers can attempt the (likely-missing) path themselves.
    return base_dir.parent / p


def load_info_with_root(data_dir, split_name):
    """Read info.csv and tag rows with their dataset root + split name."""
    import pandas as pd
    df = pd.read_csv(Path(data_dir) / "info.csv", index_col=0).reset_index(drop=True).copy()
    df["__data_root"] = str(data_dir)
    df["__source_split"] = split_name
    return df


def resolve_row_path(row, key):
    return resolve_info_path(Path(row["__data_root"]), row[key])


def remap_kaggle_paths(df, train_dir, val_dir, test_dir):
    """Rewrite path columns in a merged info.csv so each row points into the right
    kaggle-safe-renamed split dir.
    """
    import pandas as pd
    df = df.copy()
    path_cols = [*CAMERA_NAMES, *INTRINSICS_NAMES, *CAR2CAM_NAMES, GT_NAME]
    train_dir, val_dir, test_dir = Path(train_dir), Path(val_dir), Path(test_dir)
    roots = {"train": train_dir, "val": val_dir, "test": test_dir}
    anchors = [
        "autonomy_yandex_dataset_train", "autonomy_yandex_dataset_val",
        "autonomy_yandex_dataset_test",
        "autonomy_yandex_dataset_train_kaggle_safe",
        "autonomy_yandex_dataset_val_kaggle_safe",
        "autonomy_yandex_dataset_test_kaggle_safe",
    ]

    def rewrite(p, split):
        if pd.isna(p):
            return p
        pp = Path(str(p))
        if pp.exists():
            return str(pp)
        root = roots.get(str(split), train_dir)
        parts = list(pp.parts)
        for anchor in anchors:
            if anchor in parts:
                rel = Path(*parts[parts.index(anchor) + 1:])
                for cand in (root / rel, root / Path(*[kaggle_safe_name(x) for x in rel.parts])):
                    if cand.exists():
                        return str(cand)
        safe_p = Path(*[kaggle_safe_name(x) for x in parts])
        for cand in (root / safe_p, root / kaggle_safe_name(pp.name)):
            if cand.exists():
                return str(cand)
        return str(pp)

    if "__source_split" in df.columns:
        df["__data_root"] = df["__source_split"].map(
            {"train": str(train_dir), "val": str(val_dir), "test": str(test_dir)}
        ).fillna(str(train_dir))
    else:
        df["__data_root"] = str(train_dir)

    splits = df.get("__source_split", pd.Series(["train"] * len(df)))
    for col in path_cols:
        if col in df.columns:
            df[col] = [rewrite(p, s) for p, s in zip(df[col], splits)]
    return df
