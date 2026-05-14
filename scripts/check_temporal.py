"""How many distinct rides are there per split, and how big are they?"""
import pandas as pd


def check(df, name):
    rides = df.sort_values(["rover", "ride_date"]).groupby(["rover", "ride_date"]).size()
    print(f"--- {name} ---")
    print(f"samples={len(df)}  rides={len(rides)}  mean frames/ride={rides.mean():.1f}")
    print("frames/ride distribution (head):")
    print(rides.value_counts().sort_index().head(10))
    print()


if __name__ == "__main__":
    check(pd.read_csv("autonomy_yandex_dataset_train/info.csv"), "train")
    check(pd.read_csv("autonomy_yandex_dataset_test/info.csv"), "test")
