import os
import numpy as np
import pandas as pd
from typing import Dict


def create_5fold_cv(
    df: pd.DataFrame,
    output_dir: str,
    seed: int = 42
) -> None:
    """
    Create 5-fold cross-validation splits with 60/20/20
    (train/dev/test) for each fold.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (one row per essay)
    output_dir : str
        Where to save fold directories
    seed : int
        Random seed for reproducibility
    """

    os.makedirs(output_dir, exist_ok=True)

    n = len(df)
    indices = np.arange(n)

    rng = np.random.RandomState(seed)

    for fold in range(1, 6):
        print(f"Creating fold {fold}...")

        rng.shuffle(indices)

        train_end = int(0.6 * n)
        dev_end = int(0.8 * n)

        train_idx = indices[:train_end]
        dev_idx = indices[train_end:dev_end]
        test_idx = indices[dev_end:]

        train_df = df.iloc[train_idx].reset_index(drop=True)
        dev_df = df.iloc[dev_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_df.to_csv(os.path.join(fold_dir, "train.tsv"), sep="\t", index=False)
        dev_df.to_csv(os.path.join(fold_dir, "dev.tsv"), sep="\t", index=False)
        test_df.to_csv(os.path.join(fold_dir, "test.tsv"), sep="\t", index=False)

        print(
            f"  train={len(train_df)}, "
            f"dev={len(dev_df)}, "
            f"test={len(test_df)}"
        )

    print("✅ 5-fold cross-validation datasets created.")
