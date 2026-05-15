"""Train/val splits that mimic test distribution.

The official val split shares (rover, ride_date) groups with train, so its IoU
overestimates test by ~1-3 points. These splits fix that.
"""
from collections import Counter
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


def make_test_matched_split_target(info_df, test_info_csv,
                                   target_val_size=200,
                                   group_cols=("rover", "ride_date"),
                                   seed=42, cache_path=None):
    """Group-level test-matched split used by v6/v7/v8.

    Allocates val groups per-rover proportionally to test mass, then trims
    overshoot by dropping the most-redundant group of an overrepresented rover.
    """
    if cache_path is not None and Path(cache_path).exists():
        d = np.load(cache_path)
        return d["train_idx"].tolist(), d["val_idx"].tolist()

    rng = np.random.RandomState(seed)
    info = info_df.reset_index(drop=True).copy()
    test_info = pd.read_csv(test_info_csv, index_col=0).reset_index(drop=True)
    test_counts = Counter(test_info["rover"])
    test_total = max(sum(test_counts.values()), 1)

    grouped = info.groupby(list(group_cols)).groups
    rows = [
        {"group_key": k, "rover": k[0], "size": int(len(idxs))}
        for k, idxs in grouped.items()
    ]
    groups_df = pd.DataFrame(rows)
    groups_df["test_weight"] = groups_df["rover"].map(lambda r: test_counts.get(r, 0) / test_total)
    groups_df = groups_df[groups_df["test_weight"] > 0].reset_index(drop=True)

    selected = set()
    selected_rows = []

    def choose(candidate_df, target):
        candidate_df = candidate_df.sample(frac=1.0, random_state=rng.randint(0, 10**9))
        remaining = candidate_df.to_dict("records")
        chosen, total = [], 0
        while remaining and total < target:
            residual = target - total
            remaining.sort(key=lambda x: (abs(x["size"] - residual), x["size"]))
            g = remaining.pop(0)
            chosen.append(g)
            total += g["size"]
        return chosen

    for rover, _ in test_counts.most_common():
        rg = groups_df[groups_df["rover"] == rover]
        if len(rg) == 0:
            continue
        target = max(1, int(round(target_val_size * test_counts[rover] / test_total)))
        for g in choose(rg, target):
            if g["group_key"] not in selected:
                selected.add(g["group_key"])
                selected_rows.append(g)

    cur = sum(g["size"] for g in selected_rows)
    remaining_df = groups_df[~groups_df["group_key"].isin(selected)].copy()
    while cur < target_val_size and len(remaining_df) > 0:
        residual = target_val_size - cur
        remaining_df = remaining_df.sample(frac=1.0, random_state=rng.randint(0, 10**9))
        remaining_df = remaining_df.sort_values(["test_weight", "size"], ascending=[False, True])
        remaining_df = remaining_df.reset_index(drop=True)
        best = min(range(len(remaining_df)),
                   key=lambda i: (abs(int(remaining_df.iloc[i]["size"]) - residual),
                                  -float(remaining_df.iloc[i]["test_weight"])))
        g = remaining_df.iloc[best].to_dict()
        selected.add(g["group_key"])
        selected_rows.append(g)
        cur += int(g["size"])
        remaining_df = remaining_df.drop(index=best).reset_index(drop=True)

    selected_df = pd.DataFrame(selected_rows)
    while len(selected_df) > 1 and selected_df["size"].sum() > target_val_size + 20:
        overflow = int(selected_df["size"].sum() - target_val_size)
        counts = selected_df.groupby("rover").size().to_dict()
        cands = [i for i, r in selected_df.iterrows() if counts.get(r["rover"], 0) > 1]
        if not cands:
            break
        drop_i = min(cands, key=lambda i: abs(int(selected_df.loc[i, "size"]) - overflow))
        selected_df = selected_df.drop(index=drop_i).reset_index(drop=True)

    val_groups = set(selected_df["group_key"].tolist())
    train_idx, val_idx = [], []
    for i, row in info.iterrows():
        key = tuple(row[c] for c in group_cols)
        (val_idx if key in val_groups else train_idx).append(i)

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, train_idx=np.array(train_idx), val_idx=np.array(val_idx))
    return train_idx, val_idx


def build_rover_vocab_from_train(train_df, min_count=30, topk=25):
    """Build a small vocab: `__other__` for the long tail, then top-K rovers."""
    counts = train_df["rover"].value_counts()
    top = counts[counts >= min_count].head(topk)
    vocab = {"__other__": 0}
    for i, rover in enumerate(top.index.tolist(), start=1):
        vocab[rover] = i
    stats_df = pd.DataFrame({
        "rover": counts.index,
        "count": counts.values,
        "embedding_id": [vocab.get(r, 0) for r in counts.index],
        "bucket": ["unique" if r in vocab and r != "__other__" else "other" for r in counts.index],
    })
    return vocab, stats_df


def encode_rover(rover_name, rover_vocab):
    return int(rover_vocab.get(rover_name, 0))
