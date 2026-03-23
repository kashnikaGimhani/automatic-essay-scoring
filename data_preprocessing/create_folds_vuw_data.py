#!/usr/bin/env python3
import os
import argparse
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


REQUIRED_COLS = ["id", "essay", "ideas", "flow", "coherence", "vocab", "grammar", "overall_score"]


def clean_text(x: str) -> str:
    x = str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def compute_overall_from_traits(df: pd.DataFrame) -> pd.Series:
    """
    Computes overall = round(mean of trait scores).
    This matches your dataset rule and avoids relying on overall_score column.
    """
    return (
        df[["ideas", "flow", "coherence", "vocab", "grammar"]]
        .astype(float)
        .mean(axis=1)
        .round()
        .astype(int)
    )


def stratified_split_60_20_20(df: pd.DataFrame, strat_labels: np.ndarray, seed: int):
    """
    60/20/20 stratified split:
      train = 60%
      dev   = 20%
      test  = 20%
    Done as: (train vs temp=40%), then (dev vs test) from temp.

    Using 3-bin strat labels makes this stable for tiny datasets.
    """
    n = len(df)
    idx = np.arange(n)

    # First split: train(60) / temp(40)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    train_idx, temp_idx = next(sss1.split(idx, strat_labels))

    # Second split: dev(20) / test(20) from temp (50/50 of temp)
    temp_labels = strat_labels[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + 999)
    dev_rel_idx, test_rel_idx = next(sss2.split(np.arange(len(temp_idx)), temp_labels))

    dev_idx = temp_idx[dev_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[dev_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True, help="TSV/CSV with full dataset (e.g., 113 rows).")
    ap.add_argument("--sep", type=str, default="\t", help="Delimiter: '\\t' for TSV, ',' for CSV.")
    ap.add_argument("--output_dir", type=str, default="./data_small_folds", help="Output dir.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_folds", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.data_path, sep=args.sep)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nFound: {list(df.columns)}")

    # Clean essays and drop empty
    df["essay"] = df["essay"].apply(clean_text)
    df = df[df["essay"].str.len() > 0].reset_index(drop=True)

    # ---- Stratification labels (3-bin, stable) ----
    # overall = round(mean of trait scores)
    overall = compute_overall_from_traits(df).to_numpy()

    # 3 bins:
    #   0 = low (<=3)
    #   1 = mid (=4)
    #   2 = high (>=5)
    strat = np.where(overall <= 3, 0, np.where(overall == 4, 1, 2))

    # Print distributions for sanity
    print("Computed overall (from traits) distribution:")
    print(pd.Series(overall).value_counts().sort_index())
    print("\n3-bin strat distribution:")
    print(pd.Series(strat).value_counts().sort_index())
    print("")

    for fold in range(1, args.n_folds + 1):
        fold_seed = args.seed + fold * 100
        train_df, dev_df, test_df = stratified_split_60_20_20(df, strat, seed=fold_seed)

        fold_dir = os.path.join(args.output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_df.to_csv(os.path.join(fold_dir, "train.tsv"), sep="\t", index=False)
        dev_df.to_csv(os.path.join(fold_dir, "dev.tsv"), sep="\t", index=False)
        test_df.to_csv(os.path.join(fold_dir, "test.tsv"), sep="\t", index=False)

        print(f"[fold {fold}] train={len(train_df)} dev={len(dev_df)} test={len(test_df)} -> {fold_dir}")

    print("\n✅ Created 5 folds with stratified 60/20/20 train/dev/test per fold (3-bin strat).")

# python3 create_small_folds.py \
#   --data_path ../data/vuw_data/113_copy.tsv \
#   --sep $'\t' \
#   --output_dir ../data/vuw_data_folds_v2 \
#   --seed 42 \
#   --n_folds 5


if __name__ == "__main__":
    main()