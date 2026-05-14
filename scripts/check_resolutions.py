"""Per-rover sweep of image sizes, GT shapes, and focal lengths.

Reveals the rig heterogeneity (e.g. rover `nack` has a non-standard 768×959 frame
that breaks a uniform resize — motivated the letterbox dataset in v4).
"""
import numpy as np
import pandas as pd
import tqdm
from PIL import Image


def main():
    df = pd.read_csv("autonomy_yandex_dataset_train/info.csv")
    rows = []
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        with Image.open(row["/camera/inner/frontal/middle"]) as img:
            size = img.size
        gt_shape = np.load(row["gt_occupancy_grid"], mmap_mode="r").shape
        intr = np.load(row["/camera/inner/frontal/middle/intrinsic_params"])
        rows.append({
            "rover": row["rover"],
            "img_size": f"{size[0]}x{size[1]}",
            "gt_shape": str(gt_shape),
            "fx": round(intr[0, 0], 1),
            "fy": round(intr[1, 1], 1),
        })
    sdf = pd.DataFrame(rows)

    for col, title in [("img_size", "image sizes"),
                       ("gt_shape", "GT shapes"),
                       ("fx", "fx focals")]:
        print(f"\n=== {title} ===")
        counts = sdf[col].value_counts()
        print(counts)
        if col == "img_size" or col == "fx":
            for v in counts.index:
                rovers = sdf[sdf[col] == v]["rover"].unique()
                print(f"  {col}={v}: rovers={rovers[:10]} (n={len(rovers)})")


if __name__ == "__main__":
    main()
