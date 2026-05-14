"""Sanity-print a test-matched (rover, ride_date) split.

The default group-aware split gives val rover-frequencies that diverge from
test (TV ~0.35). Reweighting per rover by test mass brings it down to ~0.18.
"""
import pandas as pd

from src.splits import make_test_matched_split


def main():
    train_csv = "autonomy_yandex_dataset_train/info.csv"
    test_csv = "autonomy_yandex_dataset_test/info.csv"
    train_idx, val_idx = make_test_matched_split(train_csv, test_csv)

    n_total = len(train_idx) + len(val_idx)
    print(f"total={n_total}  train={len(train_idx)}  val={len(val_idx)} ({len(val_idx)/n_total*100:.1f}%)")

    val = pd.read_csv(train_csv, index_col=0).iloc[val_idx]
    print("\nTop val rovers:")
    print(val["rover"].value_counts().head(5))
    print("\nTop test rovers:")
    print(pd.read_csv(test_csv, index_col=0)["rover"].value_counts().head(5))


if __name__ == "__main__":
    main()
