import os
import argparse
import numpy as np
import pandas as pd

from utils import (
    ensure_dir,
    save_json,
    normalize_prompt_id,
    parse_prompt_list,
    parse_int_list,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Create reusable target few-shot splits")
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
    parser.add_argument("--dev_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def stable_prompt_offset(prompt_id: str) -> int:
    return sum(ord(c) for c in prompt_id)


def compute_split_sizes(n: int, dev_ratio: float, test_ratio: float, max_k: int) -> tuple[int, int, int]:
    if n < 3:
        raise ValueError(f"Need at least 3 samples for a target split, but got n={n}")

    n_test = max(1, int(round(n * test_ratio)))
    n_dev = max(1, int(round(n * dev_ratio)))

    while n - n_test - n_dev < max_k and (n_test > 1 or n_dev > 1):
        if n_dev >= n_test and n_dev > 1:
            n_dev -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break

    n_pool = n - n_test - n_dev
    if n_pool < max_k:
        raise ValueError(
            f"Not enough target samples. n={n}, train_pool={n_pool}, but max requested few-shot size is {max_k}. "
            "Reduce few-shot sizes or reduce dev/test ratios."
        )

    return n_pool, n_dev, n_test


def save_split(df: pd.DataFrame, path: str):
    df.to_csv(path, sep="\t", index=False)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    df = pd.read_csv(args.data_path, sep=args.sep)
    df[args.prompt_col] = df[args.prompt_col].apply(normalize_prompt_id)

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
        n_pool, n_dev, n_test = compute_split_sizes(
            n=len(target_df),
            dev_ratio=args.dev_ratio,
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
            dev_idx = perm[n_test:n_test + n_dev]
            pool_idx = perm[n_test + n_dev:]

            test_df = target_df.iloc[test_idx].copy().reset_index(drop=True)
            dev_df = target_df.iloc[dev_idx].copy().reset_index(drop=True)
            train_pool_df = target_df.iloc[pool_idx].copy().reset_index(drop=True)

            # Nested few-shot subsets: fewshot_8 is a subset of fewshot_16, etc.
            # This makes data-efficiency comparisons cleaner.
            repeat_dir = os.path.join(prompt_dir, f"repeat_{repeat_idx:02d}")
            ensure_dir(repeat_dir)

            save_split(train_pool_df, os.path.join(repeat_dir, "train_pool.tsv"))
            save_split(dev_df, os.path.join(repeat_dir, "dev.tsv"))
            save_split(test_df, os.path.join(repeat_dir, "test.tsv"))

            manifest = {
                "heldout_prompt": heldout_prompt,
                "repeat_idx": repeat_idx,
                "seed": int(repeat_seed),
                "n_target_total": int(len(target_df)),
                "n_train_pool": int(len(train_pool_df)),
                "n_dev": int(len(dev_df)),
                "n_test": int(len(test_df)),
                "train_pool_ids": train_pool_df[args.id_col].tolist() if args.id_col in train_pool_df.columns else [],
                "dev_ids": dev_df[args.id_col].tolist() if args.id_col in dev_df.columns else [],
                "test_ids": test_df[args.id_col].tolist() if args.id_col in test_df.columns else [],
                "fewshot_subsets": {},
            }

            for k in fewshot_sizes:
                fewshot_df = train_pool_df.iloc[:k].copy().reset_index(drop=True)
                fewshot_path = os.path.join(repeat_dir, f"fewshot_{k}.tsv")
                save_split(fewshot_df, fewshot_path)

                manifest["fewshot_subsets"][str(k)] = {
                    "path": os.path.basename(fewshot_path),
                    "n": int(len(fewshot_df)),
                    "ids": fewshot_df[args.id_col].tolist() if args.id_col in fewshot_df.columns else [],
                }

                summary_rows.append(
                    {
                        "heldout_prompt": heldout_prompt,
                        "repeat_idx": repeat_idx,
                        "fewshot_k": k,
                        "n_target_total": len(target_df),
                        "n_train_pool": len(train_pool_df),
                        "n_dev": len(dev_df),
                        "n_test": len(test_df),
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

# python3 create_target_fewshot_splits.py \
#   --data_path your_data.tsv \
#   --output_dir experiments/target_splits \
#   --heldout_prompts all \
#   --fewshot_sizes 8,16,32,64,128 \
#   --num_repeats 5 \
#   --dev_ratio 0.2 \
#   --test_ratio 0.2 \
#   --sep $'\t'