"""Pack test predictions into a contest-shaped zip.

The contest validator rejects `shutil.make_archive` output sometimes (saw it
with `End-of-central-directory signature not found`). Using zipfile + testzip()
plus a SHA256 readout has been reliable.
"""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from src.geometry import BEV_H, BEV_W


def write_grid(out_dir, name, grid_u8):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}.npy", grid_u8.astype(np.uint8))


def build_submission_zip(grids_dir, info_csv, out_zip, sub_folder="predicted_static_grids"):
    grids_dir = Path(grids_dir)
    out_zip = Path(out_zip)
    tmp = out_zip.with_suffix(".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    shutil.copy(info_csv, tmp / "info.csv")
    shutil.copytree(grids_dir, tmp / sub_folder)

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in tmp.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(tmp))
    shutil.rmtree(tmp)

    with zipfile.ZipFile(out_zip) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"corrupt zip entry: {bad}")
    sha = hashlib.sha256(out_zip.read_bytes()).hexdigest()
    print(f"wrote {out_zip}  size={out_zip.stat().st_size:,}  sha256={sha[:16]}...")
    return out_zip


def make_submission_from_probs(test_probs, threshold, tag, out_root,
                               test_info, info_csv, pred_name_fn,
                               name_prefix="submission", verbose=True):
    """Threshold probs, dump per-sample npy, zip with info.csv. Returns metadata dict."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    thr_tag = f"{threshold:.2f}".replace(".", "p")
    pred_dir = out_root / f"predicted_static_grids_{tag}_{thr_tag}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for p in pred_dir.glob("*.npy"):
        p.unlink()

    preds = (test_probs > threshold).numpy().astype(np.int32)
    assert len(preds) == len(test_info), (len(preds), len(test_info))
    for i, row in test_info.iterrows():
        np.save(pred_dir / pred_name_fn(row), preds[i].reshape(1, BEV_H, BEV_W))

    zip_path = out_root / f"{name_prefix}_{tag}_t_{thr_tag}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(info_csv, arcname="info.csv")
        for npy in sorted(pred_dir.glob("*.npy")):
            zf.write(npy, arcname=f"predicted_static_grids/{npy.name}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
        assert bad is None, bad

    sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    result = {
        "threshold": float(threshold),
        "tag": tag,
        "zip_path": str(zip_path.resolve()),
        "size_mb": round(zip_path.stat().st_size / 1e6, 3),
        "sha256": sha,
    }
    if verbose:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return result
