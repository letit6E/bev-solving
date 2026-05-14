"""Train/val splits that mimic test distribution.

The official val split shares (rover, ride_date) groups with train, so its IoU
overestimates test by ~1-3 points. These splits fix that.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def make_group_aware_split(info_csv, group_cols=("rover", "ride_date"),
                           holdout_frac=0.2, seed=42, cache_path=None):
    if cache_path is not None and Path(cache_path).exists():
        d = np.load(cache_path)
        return d["train_idx"].tolist(), d["val_idx"].tolist()

    info = pd.read_csv(info_csv, index_col=0).reset_index(drop=True)
    groups = list(info.groupby(list(group_cols)).groups.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    val_groups = set(groups[: max(1, int(round(len(groups) * holdout_frac)))])

    train_idx, val_idx = [], []
    for i, row in info.iterrows():
        key = tuple(row[c] for c in group_cols)
        (val_idx if key in val_groups else train_idx).append(i)

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, train_idx=np.array(train_idx), val_idx=np.array(val_idx))
    return train_idx, val_idx


def make_test_matched_split(train_info_csv, test_info_csv, target_val_size=200, seed=42):
    """Pick val samples so per-rover frequencies roughly match test distribution.

    Helped a lot once we discovered the rover `nack` was overrepresented in
    official val with mostly-empty GT (see gpt_findings_round2.md).
    """
    train = pd.read_csv(train_info_csv, index_col=0).reset_index(drop=True)
    test = pd.read_csv(test_info_csv, index_col=0)
    test_frac = test["rover"].value_counts(normalize=True).to_dict()
    rng = np.random.RandomState(seed)

    val_idx = []
    for rover, frac in test_frac.items():
        pool = train.index[train["rover"] == rover].tolist()
        if not pool:
            continue
        n_take = min(len(pool), max(1, int(round(target_val_size * frac))))
        val_idx.extend(rng.choice(pool, n_take, replace=False).tolist())
    val_idx = sorted(set(val_idx))
    train_idx = [i for i in train.index if i not in set(val_idx)]
    return train_idx, val_idx
