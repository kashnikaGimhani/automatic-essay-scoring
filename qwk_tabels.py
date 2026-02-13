import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def make_qwk_summary_tables(qwk_csv: str, out_dir: str = "qwk_tables"):
    df = pd.read_csv(qwk_csv)

    # prompt id
    if "prompt" in df.columns:
        df["prompt_id"] = df["prompt"]
    elif "Unnamed: 0" in df.columns:
        df["prompt_id"] = df["Unnamed: 0"] + 1
    else:
        df["prompt_id"] = np.arange(1, len(df) + 1)

    # metric columns
    metric_cols = [c for c in df.columns if c not in ["Unnamed: 0", "prompt", "prompt_id"]]
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # traits (exclude overall)
    traits = [c for c in metric_cols if c != "overall"]

    # Table 1: Avg QWK per trait (across prompts)
    avg_trait = (
        df[traits]
        .mean(axis=0, skipna=True)
        .to_frame(name="avg_qwk_across_prompts")
        .sort_values("avg_qwk_across_prompts", ascending=False)
        .reset_index()
        .rename(columns={"index": "trait"})
    )

    # Table 2: Avg QWK per prompt (across traits)
    avg_prompt = (
        df[["prompt_id"] + traits]
        .set_index("prompt_id")
        .mean(axis=1, skipna=True)
        .to_frame(name="avg_qwk_across_traits")
        .reset_index()
        .sort_values("prompt_id")
    )

    # optional overall column
    if "overall" in df.columns:
        overall = df.set_index("prompt_id")["overall"]
        avg_prompt["overall_qwk"] = avg_prompt["prompt_id"].map(overall)

    # save
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    avg_trait.to_csv(out_path / "avg_qwk_per_trait.csv", index=False)
    avg_prompt.to_csv(out_path / "avg_qwk_per_prompt.csv", index=False)


def csv_to_table_figure(
    csv_path,
    out_path,
    title=None,
    float_fmt="{:.3f}",
    font_size=10,
    scale=(1, 1.5)
):
    df = pd.read_csv(csv_path)

    # Format floats nicely
    df_fmt = df.copy()
    for col in df_fmt.columns:
        if pd.api.types.is_numeric_dtype(df_fmt[col]):
            df_fmt[col] = df_fmt[col].map(
                lambda x: float_fmt.format(x) if pd.notna(x) else ""
            )

    fig, ax = plt.subplots(figsize=(len(df.columns) * 1.6, len(df) * 0.4 + 1))
    ax.axis("off")

    table = ax.table(
        cellText=df_fmt.values,
        colLabels=df_fmt.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(True)
    # table.set_fontsize(font_size)
    table.scale(scale[0], scale[1])

    # Bold header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    if title:
        plt.title(title, pad=12)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    make_qwk_summary_tables(
        qwk_csv="results/fold_1/checkpoint-10000/qwk_results.csv",   # path to your CSV
        out_dir="qwk_plots"         # output folder
    )
    csv_to_table_figure(
        csv_path="qwk_plots/avg_qwk_per_trait.csv",
        out_path="qwk_plots/avg_qwk_per_trait_table.png",
        title="Average QWK Scores Across Prompts for Each Trait"
    )
    csv_to_table_figure(
        csv_path="qwk_plots/avg_qwk_per_prompt.csv",
        out_path="qwk_plots/avg_qwk_per_prompt_table.png",
        title="Average QWK Scores Across Traits for Each Prompt"
    )

