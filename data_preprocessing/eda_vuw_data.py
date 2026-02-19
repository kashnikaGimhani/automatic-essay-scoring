"""
EDA script for single-prompt / no-prompt trait-scored essay dataset.

Expected columns (example):
id, essay, ideas, flow, coherence, vocab, grammar, overall_score

Creates these plots:
1) Missingness (% missing per column)
2) Overall score distribution (hist)
3) Trait distributions (boxplot across traits)
4) Trait–overall correlation heatmap (Spearman)
5) Essay length vs overall score (scatter)
6) Essay length vs each trait (scatter; one file per trait)

Outputs: PNG files into ./eda_outputs/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Config (adjust if needed)
# ----------------------------
ID_COL = "id"
ESSAY_COL_CANDIDATES = ["essay"]
TRAITS = ["ideas", "flow", "coherence", "vocab", "grammar"]
OVERALL_COL = "overall_score"

OUT_DIR = "eda_vuw_outputs"


# ----------------------------
# Helpers
# ----------------------------
def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pick_essay_col(df: pd.DataFrame) -> str:
    for c in ESSAY_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find an essay text column. Tried: {ESSAY_COL_CANDIDATES}")


def compute_essay_length_words(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).apply(lambda s: len(s.split()))


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Convert score columns to numeric safely (handles strings).
    Non-convertible values become NaN.
    """
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ----------------------------
# Plot functions
# ----------------------------
def plot_missingness(df: pd.DataFrame, out_dir: str) -> None:
    miss_pct = (df.isna().mean() * 100).sort_values(ascending=False)

    plt.figure(figsize=(8, 4))
    plt.bar(miss_pct.index.astype(str), miss_pct.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("% missing")
    plt.title("Missingness per column")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_missingness.png"), dpi=200)
    plt.close()


def plot_overall_distribution(df: pd.DataFrame, out_dir: str) -> None:
    x = df[OVERALL_COL].dropna()

    plt.figure(figsize=(6, 4))
    plt.hist(x, bins=20)
    plt.xlabel(OVERALL_COL)
    plt.ylabel("Count")
    plt.title("Overall score distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "02_overall_score_distribution.png"), dpi=200)
    plt.close()


def plot_trait_distributions_boxplot(df: pd.DataFrame, out_dir: str) -> None:
    available_traits = [t for t in TRAITS if t in df.columns]
    if not available_traits:
        print("[WARN] No trait columns found; skipping trait distributions boxplot.")
        return

    data = [df[t].dropna().values for t in available_traits if df[t].notna().any()]
    labels = [t for t in available_traits if df[t].notna().any()]

    if not data:
        print("[WARN] Trait columns exist but are all empty; skipping.")
        return

    plt.figure(figsize=(10, 4))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Score")
    plt.title("Trait score distributions (boxplot)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "03_trait_distributions_boxplot.png"), dpi=200)
    plt.close()


def plot_trait_overall_correlation_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    available_traits = [t for t in TRAITS if t in df.columns]
    cols = available_traits + [OVERALL_COL]
    tmp = df[cols].copy()

    # Spearman correlation across numeric columns
    corr = tmp.corr(method="spearman")

    # We only want trait vs overall shown clearly
    trait_vs_overall = corr.loc[available_traits, [OVERALL_COL]].values  # shape (traits, 1)

    plt.figure(figsize=(4, max(2.5, 0.5 * len(available_traits))))
    im = plt.imshow(trait_vs_overall, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, label="Spearman correlation")
    plt.yticks(np.arange(len(available_traits)), available_traits)
    plt.xticks([0], [OVERALL_COL], rotation=0)
    plt.title("Trait–overall correlation (Spearman)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "04_trait_overall_correlation.png"), dpi=200)
    plt.close()


def plot_length_vs_overall(df: pd.DataFrame, out_dir: str) -> None:
    essay_col = pick_essay_col(df)
    df = df.copy()
    df["essay_len_words"] = compute_essay_length_words(df[essay_col])

    tmp = df[["essay_len_words", OVERALL_COL]].dropna()

    plt.figure(figsize=(6, 4))
    plt.scatter(tmp["essay_len_words"], tmp[OVERALL_COL], s=8, alpha=0.4)
    plt.xlabel("Essay length (words)")
    plt.ylabel(OVERALL_COL)
    plt.title("Essay length vs overall score")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "05_length_vs_overall_scatter.png"), dpi=200)
    plt.close()


def plot_length_vs_each_trait(df: pd.DataFrame, out_dir: str) -> None:
    essay_col = pick_essay_col(df)
    df = df.copy()
    df["essay_len_words"] = compute_essay_length_words(df[essay_col])

    for t in [x for x in TRAITS if x in df.columns]:
        tmp = df[["essay_len_words", t]].dropna()
        if len(tmp) == 0:
            continue

        plt.figure(figsize=(6, 4))
        plt.scatter(tmp["essay_len_words"], tmp[t], s=8, alpha=0.4)
        plt.xlabel("Essay length (words)")
        plt.ylabel(f"{t} score")
        plt.title(f"Essay length vs {t}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"06_length_vs_{t}.png"), dpi=200)
        plt.close()


# ----------------------------
# Main
# ----------------------------
def run_eda(input_path: str) -> None:
    ensure_outdir(OUT_DIR)

    # Load TSV/CSV based on extension
    if input_path.endswith(".tsv"):
        df = pd.read_csv(input_path, sep="\t")
    else:
        df = pd.read_csv(input_path)

    # Basic checks
    essay_col = pick_essay_col(df)
    required = [essay_col, OVERALL_COL]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: '{c}'")

    # Coerce scores to numeric
    df = coerce_numeric(df, TRAITS + [OVERALL_COL])

    # Quick summary
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("Score columns summary:\n", df[TRAITS + [OVERALL_COL]].describe())

    # Plots
    plot_missingness(df, OUT_DIR)
    plot_overall_distribution(df, OUT_DIR)
    plot_trait_distributions_boxplot(df, OUT_DIR)
    plot_trait_overall_correlation_heatmap(df, OUT_DIR)
    plot_length_vs_overall(df, OUT_DIR)
    plot_length_vs_each_trait(df, OUT_DIR)

    print(f"Saved plots to: {OUT_DIR}/")


if __name__ == "__main__":
    INPUT_FILE = "data/vuw_data/113.tsv"  # <-- change this
    run_eda(INPUT_FILE)
