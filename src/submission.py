"""Pack test predictions into a contest-shaped zip.

The contest validator rejects `shutil.make_archive` output sometimes (saw it
with `End-of-central-directory signature not found`). Using zipfile + testzip()
plus a SHA256 readout has been reliable.
"""
import hashlib
import shutil
import zipfile
from pathlib import Path

import numpy as np


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
