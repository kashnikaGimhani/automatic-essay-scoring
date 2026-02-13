import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_qwk_key_figures(qwk_csv: str, out_dir: str = "qwk_plots"):
    df = pd.read_csv(qwk_csv)

    # Prompt id (your file has 'Unnamed: 0' = 0..7)
    if "prompt" in df.columns:
        df["prompt_id"] = df["prompt"]
    elif "Unnamed: 0" in df.columns:
        df["prompt_id"] = df["Unnamed: 0"] + 1
    else:
        df["prompt_id"] = np.arange(1, len(df) + 1)

    # Metric columns (traits + overall)
    metric_cols = [c for c in df.columns if c not in ["Unnamed: 0", "prompt", "prompt_id"]]
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = {}

    # 1) Overall QWK per prompt
    if "overall" in df.columns:
        plt.figure()
        plt.bar(df["prompt_id"].astype(int), df["overall"])
        plt.xlabel("Prompt")
        plt.ylabel("QWK")
        plt.title("Overall QWK per Prompt")
        f = out_path / "overall_qwk_per_prompt.png"
        plt.savefig(f, dpi=200, bbox_inches="tight")
        plt.close()
        files["overall_qwk_per_prompt"] = str(f)

    # 2) Trait-wise heatmap (traits x prompts)
    traits = [c for c in metric_cols if c != "overall"]
    heat = df.set_index("prompt_id")[traits].T  # rows=traits, cols=prompts

    plt.figure(figsize=(max(6, 0.8 * heat.shape[1]), max(4, 0.35 * heat.shape[0] + 2)))
    im = plt.imshow(heat.values, aspect="auto")
    plt.xticks(ticks=np.arange(heat.shape[1]), labels=heat.columns.astype(int))
    plt.yticks(ticks=np.arange(heat.shape[0]), labels=heat.index)
    plt.xlabel("Prompt")
    plt.ylabel("Trait")
    plt.title("Trait-wise QWK Heatmap (Prompts × Traits)")
    plt.colorbar(im, label="QWK")
    f = out_path / "trait_qwk_heatmap.png"
    plt.savefig(f, dpi=200, bbox_inches="tight")
    plt.close()
    files["trait_qwk_heatmap"] = str(f)

    # 3) Mean QWK per trait (across prompts where that trait exists)
    trait_means = heat.mean(axis=1, skipna=True).sort_values(ascending=False)

    plt.figure(figsize=(max(6, 0.25 * len(trait_means) + 3), 4))
    plt.bar(trait_means.index, trait_means.values)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Trait")
    plt.ylabel("Mean QWK")
    plt.title("Mean QWK per Trait (Across Prompts with Labels)")
    f = out_path / "mean_qwk_per_trait.png"
    plt.savefig(f, dpi=200, bbox_inches="tight")
    plt.close()
    files["mean_qwk_per_trait"] = str(f)

    # 4) Trait spread per prompt (min/mean/max across available traits)
    prompt_trait_stats = []
    for _, row in df.iterrows():
        vals = row[traits].dropna().values.astype(float)
        if len(vals) == 0:
            prompt_trait_stats.append((row["prompt_id"], np.nan, np.nan, np.nan))
        else:
            prompt_trait_stats.append((
                row["prompt_id"],
                float(np.nanmin(vals)),
                float(np.nanmean(vals)),
                float(np.nanmax(vals))
            ))

    stats_df = pd.DataFrame(prompt_trait_stats, columns=["prompt_id","min_trait_qwk","mean_trait_qwk","max_trait_qwk"]).sort_values("prompt_id")

    plt.figure()
    x = stats_df["prompt_id"].astype(int).values
    plt.plot(x, stats_df["min_trait_qwk"].values, marker="o", label="min trait")
    plt.plot(x, stats_df["mean_trait_qwk"].values, marker="o", label="mean trait")
    plt.plot(x, stats_df["max_trait_qwk"].values, marker="o", label="max trait")
    plt.xlabel("Prompt")
    plt.ylabel("QWK")
    plt.title("Trait QWK Spread per Prompt (min/mean/max)")
    plt.legend()
    f = out_path / "trait_qwk_spread_per_prompt.png"
    plt.savefig(f, dpi=200, bbox_inches="tight")
    plt.close()
    files["trait_qwk_spread_per_prompt"] = str(f)

    return files

if __name__ == "__main__":
    files = plot_qwk_key_figures(
        qwk_csv="results/fold_1/checkpoint-10000/qwk_results.csv",
        out_dir="qwk_plots"
    )

