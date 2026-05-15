"""Dataset cleaning: empty-GT removal + near-stationary deduplication.

Used by v6/v7/v8 notebooks. Caches results so the (slow) hash + GT scan only
runs once per dataset directory.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.geometry import GT_NAME, load_info_with_root, resolve_row_path


def build_img_hash(path):
    img = Image.open(path).convert("L").resize((32, 32), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def compute_gt_stats(info_df, cache_csv=None):
    if cache_csv is not None and Path(cache_csv).exists():
        stats = pd.read_csv(cache_csv)
        if len(stats) == len(info_df):
            return stats

    rows = []
    for i, row in tqdm(info_df.iterrows(), total=len(info_df), desc="GT stats"):
        gt = np.load(resolve_row_path(row, GT_NAME)).squeeze()
        gt = np.where(gt < 0, 255, gt)
        valid = gt != 255
        rows.append({
            "__row_id": int(i),
            "coverage": float(valid.mean()),
            "valid_count": int(valid.sum()),
            "pos_count": int((gt == 1).sum()),
        })
    stats = pd.DataFrame(rows)
    if cache_csv is not None:
        stats.to_csv(cache_csv, index=False)
    return stats


def smart_deduplicate(info_df, mae_thr=0.02, camera_name="/camera/inner/frontal/middle"):
    """Group consecutive same-rover/ride frames whose front-camera hash MAE is below
    `mae_thr` and keep the frame with the richest GT (most positives)."""
    info_sorted = info_df.sort_values(["rover", "ride_date", "message_ts"]).reset_index(drop=False)
    hash_cache = {}
    keep_ids = []
    dup_groups = []

    def get_hash(idx, row):
        if idx not in hash_cache:
            hash_cache[idx] = build_img_hash(resolve_row_path(row, camera_name))
        return hash_cache[idx]

    def flush(cluster):
        if not cluster:
            return
        if len(cluster) == 1:
            keep_ids.append(cluster[0]["orig_row_idx"])
            return
        best = sorted(
            cluster,
            key=lambda x: (-x["pos_count"], -x["valid_count"], x["message_ts"]),
        )[0]
        keep_ids.append(best["orig_row_idx"])
        dup_groups.append({
            "kept_row_id": int(best["orig_row_idx"]),
            "group_size": int(len(cluster)),
            "members": [int(x["orig_row_idx"]) for x in cluster],
        })

    for (_, _), sub in tqdm(info_sorted.groupby(["rover", "ride_date"], sort=False),
                            desc="dedup groups"):
        records = []
        for _, r in sub.iterrows():
            idx = int(r["index"])
            ref = info_df.iloc[idx]
            records.append({
                "orig_row_idx": idx,
                "row": ref,
                "pos_count": int(ref["pos_count"]),
                "valid_count": int(ref["valid_count"]),
                "message_ts": str(ref.get("message_ts", "")),
            })
        if not records:
            continue
        cluster = [records[0]]
        prev = get_hash(records[0]["orig_row_idx"], records[0]["row"])
        for rec in records[1:]:
            cur = get_hash(rec["orig_row_idx"], rec["row"])
            if float(np.mean(np.abs(cur - prev))) < mae_thr:
                cluster.append(rec)
            else:
                flush(cluster)
                cluster = [rec]
            prev = cur
        flush(cluster)

    keep_ids = sorted(set(keep_ids))
    return info_df.iloc[keep_ids].reset_index(drop=True).copy(), pd.DataFrame(dup_groups)


def clean_merged_info(train_dir, val_dir, cache_dir,
                      mae_thr=0.02, dedup_camera="/camera/inner/frontal/middle",
                      use_cache=True):
    """Merge train+val, drop empty-GT rows, dedup near-stationary clusters. Caches
    the result so repeat runs are instant."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    merged_cache = cache_dir / "merged_cleaned.csv"
    stats_cache = cache_dir / "merged_gt_stats.csv"
    dup_cache = cache_dir / "dedup_report.csv"
    summary_path = cache_dir / "clean_summary.json"

    if use_cache and merged_cache.exists() and summary_path.exists():
        info = pd.read_csv(merged_cache)
        dup_df = pd.read_csv(dup_cache) if dup_cache.exists() else pd.DataFrame()
        return info, dup_df, json.loads(summary_path.read_text())

    merged = pd.concat([
        load_info_with_root(train_dir, "train"),
        load_info_with_root(val_dir, "val"),
    ], ignore_index=True)
    stats = compute_gt_stats(merged, cache_csv=stats_cache)
    merged = merged.join(stats[["coverage", "valid_count", "pos_count"]])
    before = len(merged)
    merged = merged[merged["pos_count"] > 0].reset_index(drop=True).copy()
    after_empty = len(merged)
    deduped, dup_df = smart_deduplicate(merged, mae_thr=mae_thr, camera_name=dedup_camera)

    deduped.to_csv(merged_cache, index=False)
    dup_df.to_csv(dup_cache, index=False)
    summary = {
        "merged_before_clean": int(before),
        "removed_empty_gt": int(before - after_empty),
        "after_empty_filter": int(after_empty),
        "removed_by_dedup": int(after_empty - len(deduped)),
        "clean_total": int(len(deduped)),
        "dedup_groups": int(len(dup_df)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return deduped, dup_df, summary
