import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils import (
    ensure_dir,
    save_json,
    normalize_prompt_id,
    parse_prompt_list,
    parse_int_list,
)


BALANCE_LABEL_COL = "__balance_label__"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create reusable target few-shot splits with a separate balanced dev set for each k-shot setting"
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sep", type=str, default="\t")

    parser.add_argument("--prompt_col", type=str, default="essay_set")
    parser.add_argument("--id_col", type=str, default="essay_id")

    parser.add_argument(
        "--heldout_prompts",
        type=str,
        default="all",
        help='Comma list like "1,2,3" or "all"',
    )
    parser.add_argument(
        "--fewshot_sizes",
        type=str,
        default="8,16,32,64,128",
        help='Comma list like "8,16,32,64"',
    )
    parser.add_argument("--num_repeats", type=int, default=5)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--balance_col",
        type=str,
        default="overall_score",
        help=(
            "Column used to balance each dev_k set. Default: overall_score. "
            "If this column is not available, provide --balance_trait_cols."
        ),
    )
    parser.add_argument(
        "--balance_trait_cols",
        type=str,
        default="",
        help=(
            "Optional comma-separated trait columns used to compute a rounded mean balance label "
            "when --balance_col is not present, e.g. content,organization,word_choice,sentence_fluency,conventions."
        ),
    )
    parser.add_argument(
        "--balance_bins",
        type=int,
        default=0,
        help=(
            "Optional number of quantile bins for balancing. Use 0 to balance by exact score labels. "
            "For sparse scores, 3 bins is often safer than exact score balancing."
        ),
    )

    return parser.parse_args()


def stable_prompt_offset(prompt_id: str) -> int:
    return sum(ord(c) for c in prompt_id)


def parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def compute_test_size(n: int, test_ratio: float, max_k: int) -> int:
    """
    We need enough non-test examples for the largest k-shot setting:
        max_k train examples + max_k dev examples.

    This script creates dev_k with size k, so the minimum required pool is 2 * max_k.
    """
    if n < 3:
        raise ValueError(f"Need at least 3 samples for a target split, but got n={n}")

    n_test = max(1, int(round(n * test_ratio)))
    required_pool = 2 * max_k

    while n - n_test < required_pool and n_test > 1:
        n_test -= 1

    if n - n_test < required_pool:
        raise ValueError(
            f"Not enough target samples for per-k dev sets. n={n}, available_after_test={n - n_test}, "
            f"but the largest requested few-shot size is {max_k}, requiring at least {2 * max_k} "
            "non-test samples: max_k train + max_k dev. Reduce few-shot sizes, reduce test_ratio, "
            "or use a smaller validation budget."
        )

    return n_test


def make_balance_label(df: pd.DataFrame, args) -> pd.Series:
    """Return the label used to balance dev_k sets."""
    if args.balance_col and args.balance_col in df.columns:
        raw = pd.to_numeric(df[args.balance_col], errors="coerce")
    else:
        trait_cols = parse_str_list(args.balance_trait_cols)
        if not trait_cols:
            raise ValueError(
                f"Balance column '{args.balance_col}' was not found. Either set --balance_col to an existing "
                "score column, or provide --balance_trait_cols to compute rounded mean score labels."
            )
        missing = [c for c in trait_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns from --balance_trait_cols: {missing}")
        raw = df[trait_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).round()

    if raw.isna().any():
        bad_n = int(raw.isna().sum())
        raise ValueError(
            f"Found {bad_n} missing/non-numeric values in the balance labels. "
            "Check --balance_col or --balance_trait_cols."
        )

    if args.balance_bins and args.balance_bins > 1 and raw.nunique() > args.balance_bins:
        # qcut gives roughly equal-frequency bins. duplicates='drop' avoids errors if many repeated values exist.
        labels = pd.qcut(raw, q=args.balance_bins, duplicates="drop").astype(str)
        return labels

    # Use clean score-like labels, e.g. 1,2,3 instead of 1.0,2.0,3.0 when possible.
    if np.allclose(raw.values, np.round(raw.values)):
        return raw.round().astype(int).astype(str)
    return raw.astype(str)


def label_counts(df: pd.DataFrame, label_col: str = BALANCE_LABEL_COL) -> Dict[str, int]:
    if label_col not in df.columns:
        return {}
    counts = df[label_col].value_counts().sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def balanced_sample_indices(
    df: pd.DataFrame,
    k: int,
    rng: np.random.RandomState,
    label_col: str = BALANCE_LABEL_COL,
) -> List[int]:
    """
    Sample k rows in an approximately balanced way across score labels.

    Logic:
    1. Give each label floor(k / number_of_labels) examples.
    2. Distribute the remaining examples one-by-one across labels.
    3. If a label does not have enough examples, reallocate the leftover quota to labels with capacity.
    """
    if len(df) < k:
        raise ValueError(f"Cannot sample k={k} from only {len(df)} candidate rows")
    if label_col not in df.columns:
        raise ValueError(f"Missing required balance label column: {label_col}")

    labels = sorted(df[label_col].astype(str).unique().tolist())
    if not labels:
        raise ValueError("No labels found for balanced sampling")

    # Shuffle label order so remainder allocation is not always biased toward low/high labels.
    labels = labels.copy()
    rng.shuffle(labels)

    group_indices: Dict[str, List[int]] = {}
    for label in labels:
        idx = df.index[df[label_col].astype(str) == label].tolist()
        rng.shuffle(idx)
        group_indices[label] = idx

    base = k // len(labels)
    remainder = k % len(labels)

    quotas = {label: base for label in labels}
    for label in labels[:remainder]:
        quotas[label] += 1

    selected_by_label: Dict[str, List[int]] = {}
    leftover = 0

    for label in labels:
        available = len(group_indices[label])
        take = min(quotas[label], available)
        selected_by_label[label] = group_indices[label][:take]
        leftover += quotas[label] - take

    # Reallocate leftover quota to labels that still have unused examples.
    while leftover > 0:
        candidates = [
            label for label in labels
            if len(selected_by_label[label]) < len(group_indices[label])
        ]
        if not candidates:
            break
        rng.shuffle(candidates)
        for label in candidates:
            if leftover <= 0:
                break
            current_n = len(selected_by_label[label])
            selected_by_label[label].append(group_indices[label][current_n])
            leftover -= 1

    selected: List[int] = []
    for label in labels:
        selected.extend(selected_by_label[label])

    if len(selected) != k:
        raise ValueError(
            f"Balanced sampling failed: requested k={k}, selected {len(selected)}. "
            f"Candidate pool size={len(df)}."
        )

    rng.shuffle(selected)
    return selected


def save_split(df: pd.DataFrame, path: str):
    # Remove internal helper columns before saving actual splits.
    out_df = df.drop(columns=[BALANCE_LABEL_COL], errors="ignore")
    out_df.to_csv(path, sep="\t", index=False)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    df = pd.read_csv(args.data_path, sep=args.sep)
    df[args.prompt_col] = df[args.prompt_col].apply(normalize_prompt_id)
    df[BALANCE_LABEL_COL] = make_balance_label(df, args)

    all_prompts = sorted(df[args.prompt_col].astype(str).unique().tolist())
    heldout_prompts = parse_prompt_list(args.heldout_prompts, all_prompts)
    fewshot_sizes = sorted(parse_int_list(args.fewshot_sizes))

    summary_rows = []

    for heldout_prompt in heldout_prompts:
        target_df = df[df[args.prompt_col] == heldout_prompt].copy().reset_index(drop=True)

        if len(target_df) == 0:
            print(f"Skipping heldout prompt {heldout_prompt}: no rows found")
            continue

        max_k = max(fewshot_sizes)
        n_test = compute_test_size(
            n=len(target_df),
            test_ratio=args.test_ratio,
            max_k=max_k,
        )

        prompt_dir = os.path.join(args.output_dir, f"heldout_{heldout_prompt}")
        ensure_dir(prompt_dir)

        for repeat_idx in range(1, args.num_repeats + 1):
            repeat_seed = args.seed + 1000 * repeat_idx + stable_prompt_offset(heldout_prompt)
            rng = np.random.RandomState(repeat_seed)

            perm = rng.permutation(len(target_df))
            test_idx = perm[:n_test]
            candidate_idx = perm[n_test:]

            test_df = target_df.iloc[test_idx].copy().reset_index(drop=True)
            candidate_pool_df = target_df.iloc[candidate_idx].copy().reset_index(drop=True)

            # Nested few-shot train sets:
            # fewshot_8 is a subset of fewshot_16, fewshot_16 is a subset of fewshot_32, etc.
            # For each k, dev_k is sampled from rows AFTER the first k training rows,
            # so fewshot_k and dev_k are always disjoint.
            repeat_dir = os.path.join(prompt_dir, f"repeat_{repeat_idx:02d}")
            ensure_dir(repeat_dir)

            save_split(candidate_pool_df, os.path.join(repeat_dir, "candidate_pool.tsv"))
            save_split(test_df, os.path.join(repeat_dir, "test.tsv"))

            manifest = {
                "heldout_prompt": heldout_prompt,
                "repeat_idx": repeat_idx,
                "seed": int(repeat_seed),
                "balance_col": args.balance_col,
                "balance_trait_cols": parse_str_list(args.balance_trait_cols),
                "balance_bins": int(args.balance_bins),
                "n_target_total": int(len(target_df)),
                "n_candidate_pool": int(len(candidate_pool_df)),
                "n_test": int(len(test_df)),
                "candidate_pool_ids": candidate_pool_df[args.id_col].tolist() if args.id_col in candidate_pool_df.columns else [],
                "test_ids": test_df[args.id_col].tolist() if args.id_col in test_df.columns else [],
                "test_balance_counts": label_counts(test_df),
                "fewshot_subsets": {},
            }

            for k in fewshot_sizes:
                fewshot_df = candidate_pool_df.iloc[:k].copy().reset_index(drop=True)

                dev_candidate_df = candidate_pool_df.iloc[k:].copy().reset_index(drop=True)
                dev_indices = balanced_sample_indices(dev_candidate_df, k=k, rng=rng)
                dev_df = dev_candidate_df.loc[dev_indices].copy().reset_index(drop=True)

                fewshot_path = os.path.join(repeat_dir, f"fewshot_{k}.tsv")
                dev_path = os.path.join(repeat_dir, f"dev_{k}.tsv")

                save_split(fewshot_df, fewshot_path)
                save_split(dev_df, dev_path)

                fewshot_ids = fewshot_df[args.id_col].tolist() if args.id_col in fewshot_df.columns else []
                dev_ids = dev_df[args.id_col].tolist() if args.id_col in dev_df.columns else []

                if args.id_col in fewshot_df.columns:
                    overlap = sorted(set(fewshot_ids).intersection(set(dev_ids)))
                    if overlap:
                        raise RuntimeError(
                            f"Overlap found between fewshot_{k} and dev_{k} for heldout={heldout_prompt}, "
                            f"repeat={repeat_idx}: {overlap[:10]}"
                        )

                manifest["fewshot_subsets"][str(k)] = {
                    "fewshot_path": os.path.basename(fewshot_path),
                    "dev_path": os.path.basename(dev_path),
                    "fewshot_n": int(len(fewshot_df)),
                    "dev_n": int(len(dev_df)),
                    "fewshot_ids": fewshot_ids,
                    "dev_ids": dev_ids,
                    "fewshot_balance_counts": label_counts(fewshot_df),
                    "dev_balance_counts": label_counts(dev_df),
                }

                summary_rows.append(
                    {
                        "heldout_prompt": heldout_prompt,
                        "repeat_idx": repeat_idx,
                        "fewshot_k": k,
                        "n_target_total": len(target_df),
                        "n_candidate_pool": len(candidate_pool_df),
                        "n_fewshot_train": len(fewshot_df),
                        "n_dev": len(dev_df),
                        "n_test": len(test_df),
                        "fewshot_balance_counts": label_counts(fewshot_df),
                        "dev_balance_counts": label_counts(dev_df),
                        "test_balance_counts": label_counts(test_df),
                    }
                )

            save_json(manifest, os.path.join(repeat_dir, "split_manifest.json"))
            print(f"Saved splits: heldout={heldout_prompt}, repeat={repeat_idx:02d}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(args.output_dir, "split_summary.csv"), index=False)

    print("Done.")


if __name__ == "__main__":
    main()


# Example 1: balance dev sets using an existing overall_score column
# python3 create_fewshot_splits_v2.py \
#   --data_path ../data/asap_train_with_all_traits.tsv \
#   --output_dir target_splits_v2 \
#   --heldout_prompts 2,4,6 \
#   --fewshot_sizes 8,16,32,64,128 \
#   --num_repeats 5 \
#   --test_ratio 0.2 \
#   --balance_col overall \
#   --sep $'\t'

# Example 2: if no overall_score column exists, balance by rounded mean of trait scores
# python3 create_fewshot_splits_balanced_dev.py \
#   --data_path ../data/asap_train_with_all_traits.tsv \
#   --output_dir target_splits_balanced_dev \
#   --heldout_prompts 2,4,6 \
#   --fewshot_sizes 8,16,32,64,128 \
#   --num_repeats 5 \
#   --test_ratio 0.2 \
#   --balance_trait_cols content,organization,word_choice,sentence_fluency,conventions \
#   --sep $'\t'
