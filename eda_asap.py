"""
EDA script for the combined ASAP + ASAP++ traits dataset.

This script creates these EDA plots:
1) Essay count per prompt
2) Trait availability heatmap (% non-null per prompt)
3) Overall score distribution per prompt (boxplot)
4) Trait–overall correlation heatmap (per prompt)
5) Essay length vs overall score (scatter)

Outputs: PNG files into ./eda_outputs/
"""

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Config
# ----------------------------
UNIFIED_TRAITS = [
    "content",
    "organization",
    "word_choice",
    "sentence_fluency",
    "conventions",
    "prompt_adherence",
    "language",
    "narrativity",
    "style",
    "voice",
]

PROMPT_COL = "essay_set"       
TEXT_COL_CANDIDATES = ["essay", "content_text"]
OVERALL_COL = "score_domain1" 

OUT_DIR = "eda_outputs"


# ----------------------------
# Helpers
# ----------------------------
def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pick_text_col(df: pd.DataFrame) -> str:
    for c in TEXT_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find a text column. Tried: {TEXT_COL_CANDIDATES}")


def safe_parse_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional: if you stored traits inside a 'target' dict/string and trait columns are missing,
    this will try to expand it.
    - If trait columns already exist, it does nothing.
    """
    if all(t in df.columns for t in UNIFIED_TRAITS) and OVERALL_COL in df.columns:
        return df

    if "target" not in df.columns:
        return df

    def parse_obj(x):
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            # handle "nan" inside dict-strings by converting to None
            # Example: "{'voice': 8.0, 'style': nan, ...}"
            x2 = x.replace("nan", "None")
            try:
                return ast.literal_eval(x2)
            except Exception:
                return None
        return None

    targets = df["target"].apply(parse_obj)
    # Expand into columns
    expanded = pd.json_normalize(targets)
    # Bring into df
    for t in UNIFIED_TRAITS:
        if t in expanded.columns and t not in df.columns:
            df[t] = expanded[t]
    if "overall" in expanded.columns and OVERALL_COL not in df.columns:
        df[OVERALL_COL] = expanded["overall"]

    return df


def compute_essay_length(series: pd.Series) -> pd.Series:
    # word count (simple + robust)
    return series.fillna("").astype(str).apply(lambda s: len(s.split()))


# ----------------------------
# Plot functions
# ----------------------------
def plot_count_per_prompt(df: pd.DataFrame, out_dir: str) -> None:
    counts = df[PROMPT_COL].value_counts().sort_index()

    plt.figure()
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("Prompt (essay_set)")
    plt.ylabel("Number of essays")
    plt.title("Essay count per prompt")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_count_per_prompt.png"), dpi=200)
    plt.close()


def plot_trait_availability_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    prompts = sorted(df[PROMPT_COL].dropna().unique().astype(int).tolist())

    avail = []
    for p in prompts:
        sub = df[df[PROMPT_COL] == p]
        row = []
        for t in UNIFIED_TRAITS:
            if t in sub.columns:
                row.append(100.0 * sub[t].notna().mean())
            else:
                row.append(0.0)
        avail.append(row)

    mat = np.array(avail)

    plt.figure(figsize=(12, 4))
    ax = plt.gca()

    im = ax.imshow(mat, aspect="auto")
    plt.colorbar(im, label="% non-null")

    # Axis labels
    ax.set_xticks(np.arange(len(UNIFIED_TRAITS)))
    ax.set_xticklabels(UNIFIED_TRAITS, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(prompts)))
    ax.set_yticklabels([str(p) for p in prompts])

    ax.set_xlabel("Traits")
    ax.set_ylabel("Prompt")
    ax.set_title("Trait availability heatmap (% non-null per prompt)")

    # ---- ADD CELL BORDERS ----
    ax.set_xticks(np.arange(-0.5, len(UNIFIED_TRAITS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(prompts), 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "02_trait_availability_heatmap.png"), dpi=200)
    plt.close()

def plot_overall_score_by_prompt_boxplot(df: pd.DataFrame, out_dir: str) -> None:
    prompts = sorted(df[PROMPT_COL].dropna().unique().astype(int).tolist())
    data = []
    labels = []
    for p in prompts:
        sub = df[df[PROMPT_COL] == p][OVERALL_COL].dropna()
        if len(sub) > 0:
            data.append(sub.values)
            labels.append(str(p))

    plt.figure(figsize=(10, 4))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.xlabel("Prompt (essay_set)")
    plt.ylabel("Overall score (score_domain1)")
    plt.title("Overall score distribution per prompt (boxplot)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "03_overall_score_boxplot.png"), dpi=200)
    plt.close()


def plot_trait_overall_correlation_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    """
    Correlation per prompt: corr(trait, overall) using Spearman (rank-based, good for ordinal scores).
    Produces a heatmap: rows=prompts, cols=traits
    """
    prompts = sorted(df[PROMPT_COL].dropna().unique().astype(int).tolist())
    corr_rows = []

    for p in prompts:
        sub = df[df[PROMPT_COL] == p].copy()
        row = []
        for t in UNIFIED_TRAITS:
            if t not in sub.columns:
                row.append(np.nan)
                continue
            tmp = sub[[t, OVERALL_COL]].dropna()
            if len(tmp) < 3:
                row.append(np.nan)
            else:
                row.append(tmp[t].corr(tmp[OVERALL_COL], method="spearman"))
        corr_rows.append(row)

    mat = np.array(corr_rows)

    plt.figure(figsize=(12, 4))
    im = plt.imshow(mat, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, label="Spearman correlation")

    plt.xticks(np.arange(len(UNIFIED_TRAITS)), UNIFIED_TRAITS, rotation=45, ha="right")
    plt.yticks(np.arange(len(prompts)), [str(p) for p in prompts])
    plt.xlabel("Traits")
    plt.ylabel("Prompt")
    plt.title("Trait–overall correlation (Spearman) per prompt")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "04_trait_overall_correlation_heatmap.png"), dpi=200)
    plt.close()


def plot_length_vs_overall(df: pd.DataFrame, out_dir: str) -> None:
    text_col = pick_text_col(df)
    df = df.copy()
    df["essay_len_words"] = compute_essay_length(df[text_col])

    tmp = df[[PROMPT_COL, "essay_len_words", OVERALL_COL]].dropna()

    plt.figure(figsize=(6, 4))
    plt.scatter(tmp["essay_len_words"], tmp[OVERALL_COL], s=6, alpha=0.4)
    plt.xlabel("Essay length (words)")
    plt.ylabel("Overall score (score_domain1)")
    plt.title("Essay length vs overall score")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "05_length_vs_overall_scatter.png"), dpi=200)
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

    # If traits were stored inside a target column, try to expand
    df = safe_parse_target_column(df)

    # Basic checks
    if PROMPT_COL not in df.columns:
        raise ValueError(f"Missing prompt column '{PROMPT_COL}'")
    if OVERALL_COL not in df.columns:
        raise ValueError(f"Missing overall score column '{OVERALL_COL}'")

    # Make plots
    plot_count_per_prompt(df, OUT_DIR)
    plot_trait_availability_heatmap(df, OUT_DIR)
    plot_overall_score_by_prompt_boxplot(df, OUT_DIR)
    plot_trait_overall_correlation_heatmap(df, OUT_DIR)
    plot_length_vs_overall(df, OUT_DIR)
    plot_same_trait_distribution_across_prompts(df, "content", OUT_DIR)
    plot_same_trait_distribution_across_prompts(df, "organization", OUT_DIR)
    plot_same_trait_distribution_across_prompts(df, "prompt_adherence", OUT_DIR)
    plot_same_trait_distribution_across_prompts(df, "narrativity", OUT_DIR)
    plot_same_trait_distribution_across_prompts(df, "style", OUT_DIR)
    plot_same_trait_distribution_across_prompts(df, "voice", OUT_DIR)


    print(f"Saved plots to: {OUT_DIR}/")

def plot_same_trait_distribution_across_prompts(
    df: pd.DataFrame,
    trait: str,
    out_dir: str,
) -> None:
    """
    Boxplot of a single trait's distribution across prompts.

    x-axis: prompt (essay_set)
    y-axis: trait score

    Shows that the same trait has different distributions across prompts,
    motivating prompt-aware / meta-learning approaches.
    """
    prompts = sorted(df[PROMPT_COL].dropna().unique().astype(int).tolist())

    data = []
    labels = []

    for p in prompts:
        sub = df[(df[PROMPT_COL] == p) & (df[trait].notna())]
        if len(sub) > 0:
            data.append(sub[trait].values)
            labels.append(str(p))

    if len(data) == 0:
        print(f"[WARN] No data found for trait '{trait}'. Skipping plot.")
        return

    plt.figure(figsize=(10, 4))
    plt.boxplot(
        data,
        tick_labels=labels,   # matplotlib >= 3.9
        showfliers=False
    )

    plt.xlabel("Prompt")
    plt.ylabel(f"{trait} score")
    plt.title(f"Distribution of '{trait}' scores across prompts")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"same_trait_distribution_{trait}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved: {out_path}")



if __name__ == "__main__":
    INPUT_FILE = "data/asap_train_with_all_traits.tsv"
    run_eda(INPUT_FILE)
