"""Group consecutive same-rover frames that look identical (frontal MAE<0.02) and
keep one — the one with the most obstacle pixels in its GT.

Trims a few hundred near-stationary duplicates that just inflate trivial samples.
"""
import numpy as np
import pandas as pd
import tqdm
from PIL import Image

INFO = "autonomy_yandex_dataset_train/info.csv"
OUT = "autonomy_yandex_dataset_train/info_smart_dedup.csv"


def img_hash(path):
    img = Image.open(path).convert("L").resize((32, 32))
    return np.array(img, dtype=np.float32) / 255.0


def main():
    df = pd.read_csv(INFO).sort_values(["rover", "ride_date", "message_ts"]).reset_index(drop=True)

    groups, cur = [], [0]
    for i in tqdm.tqdm(range(1, len(df)), desc="grouping"):
        same = (df.loc[i, "rover"] == df.loc[i - 1, "rover"]
                and df.loc[i, "ride_date"] == df.loc[i - 1, "ride_date"])
        if same:
            mae = np.mean(np.abs(
                img_hash(df.loc[i - 1, "/camera/inner/frontal/middle"]) -
                img_hash(df.loc[i, "/camera/inner/frontal/middle"])))
            if mae < 0.02:
                cur.append(i)
                continue
        groups.append(cur)
        cur = [i]
    groups.append(cur)

    kept = []
    for g in tqdm.tqdm(groups, desc="picking best"):
        if len(g) == 1:
            kept.append(g[0])
            continue
        best, best_obs = g[0], -1
        for idx in g:
            obs = (np.load(df.loc[idx, "gt_occupancy_grid"])[0] == 1).sum()
            if obs > best_obs:
                best, best_obs = idx, obs
        kept.append(best)

    print(f"groups={len(groups)}  dropped={len(df) - len(kept)}  kept={len(kept)}")
    df.iloc[kept].reset_index(drop=True).to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
