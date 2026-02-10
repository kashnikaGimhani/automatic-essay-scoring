import pandas as pd

# ---- CONFIG ----
INPUT_TSV = "data/5fold_cv/fold_1/test.tsv"
OUTPUT_TSV = "data/5fold_cv/fold_1/test_subset.tsv"
N_SAMPLES = 125
RANDOM_SEED = 42  # ensures reproducibility
# ----------------

def main():
    # Load full test TSV
    df = pd.read_csv(INPUT_TSV, sep="\t")

    # Safety check
    if len(df) < N_SAMPLES:
        raise ValueError(f"Test set has only {len(df)} rows, cannot sample {N_SAMPLES}")

    # Randomly sample rows
    subset_df = df.sample(n=N_SAMPLES, random_state=RANDOM_SEED)

    # Save as TSV
    subset_df.to_csv(OUTPUT_TSV, sep="\t", index=False)

    print(f"Saved {N_SAMPLES} random samples to:")
    print(OUTPUT_TSV)

if __name__ == "__main__":
    main()
