"""Sanity-check: how stable is the GT between near-identical consecutive frames?

If the rover is stationary the GT should be essentially constant. A low IoU
here means the labels themselves are noisy — caps the achievable test IoU.
"""
import numpy as np
import pandas as pd
import tqdm
from PIL import Image


def img_hash(p):
    return np.array(Image.open(p).convert("L").resize((32, 32)), dtype=np.float32) / 255.0


def iou(gt1, gt2):
    a, b = gt1 == 1, gt2 == 1
    u = (a | b).sum()
    return 1.0 if u == 0 else (a & b).sum() / u


def main():
    df = pd.read_csv("autonomy_yandex_dataset_train/info.csv")
    df = df.sort_values(["rover", "ride_date", "message_ts"]).reset_index(drop=True)
    ious = []
    for i in tqdm.tqdm(range(1, len(df))):
        if (df.loc[i, "rover"] != df.loc[i - 1, "rover"]
                or df.loc[i, "ride_date"] != df.loc[i - 1, "ride_date"]):
            continue
        mae = np.mean(np.abs(
            img_hash(df.loc[i, "/camera/inner/frontal/middle"]) -
            img_hash(df.loc[i - 1, "/camera/inner/frontal/middle"])))
        if mae < 0.02:
            ious.append(iou(
                np.load(df.loc[i, "gt_occupancy_grid"])[0],
                np.load(df.loc[i - 1, "gt_occupancy_grid"])[0]))
    if not ious:
        print("no near-stationary pairs found")
        return
    print(f"pairs={len(ious)}  mean={np.mean(ious):.3f}  median={np.median(ious):.3f}")
    for t in (0.95, 0.90, 0.80, 0.50):
        cmp = ">" if t > 0.5 else "<"
        n = sum(1 for x in ious if (x > t if cmp == ">" else x < t))
        print(f"  IoU {cmp} {t}: {n}")


if __name__ == "__main__":
    main()
