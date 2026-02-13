import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

def _read_tb_csv(path: str) -> pd.DataFrame:
    """
    Reads a TensorBoard-exported scalar CSV robustly and returns:
    columns: step, value
    """
    df = pd.read_csv(path)

    cols = {c.lower(): c for c in df.columns}

    # Find step column
    step_col = None
    for cand in ["step", "steps", "global_step", "globalstep"]:
        if cand in cols:
            step_col = cols[cand]
            break
    if step_col is None:
        step_col = df.select_dtypes(include=["number"]).columns[0]

    # Find value column
    value_col = None
    for cand in ["value", "scalar", "val"]:
        if cand in cols:
            value_col = cols[cand]
            break
    if value_col is None:
        num_cols = list(df.select_dtypes(include=["number"]).columns)
        value_col = num_cols[1] if len(num_cols) > 1 else num_cols[0]

    out = df[[step_col, value_col]].rename(columns={step_col: "step", value_col: "value"}).copy()
    out = out.dropna(subset=["step", "value"]).sort_values("step").reset_index(drop=True)
    return out

def plot_training_eval_curves(
    train_loss_csv: str,
    eval_loss_csv: str,
    *,
    title_prefix: str = "",
    smooth_window: Optional[int] = None,
    save_dir: Optional[str] = None
):
    """
    Plots:
      1) Training Loss vs step
      2) Validation Loss vs step
      3) Overlay: Train vs Validation Loss

    Args:
      train_loss_csv: path to train loss CSV
      eval_loss_csv: path to eval loss CSV
      title_prefix: optional prefix like "Fold 1 - "
      smooth_window: optional moving average window (points). None disables smoothing.
      save_dir: if provided, saves PNGs and returns file paths.

    Returns:
      dict with dataframes and (if saving) PNG paths.
    """
    train_df = _read_tb_csv(train_loss_csv)
    eval_df  = _read_tb_csv(eval_loss_csv)

    def smooth(series: pd.Series) -> pd.Series:
        if smooth_window is None or smooth_window <= 1:
            return series
        return series.rolling(window=smooth_window, min_periods=1).mean()

    train_y = smooth(train_df["value"])
    eval_y  = smooth(eval_df["value"])

    files = {}
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Train loss
        plt.figure()
        plt.plot(train_df["step"], train_y)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"{title_prefix}Training Loss")
        f1 = save_path / "train_loss.png"
        plt.savefig(f1, dpi=200, bbox_inches="tight")
        plt.close()
        files["train_loss"] = str(f1)

        # Eval loss
        plt.figure()
        plt.plot(eval_df["step"], eval_y)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"{title_prefix}Validation Loss")
        f2 = save_path / "eval_loss.png"
        plt.savefig(f2, dpi=200, bbox_inches="tight")
        plt.close()
        files["eval_loss"] = str(f2)

        # Overlay
        plt.figure()
        plt.plot(train_df["step"], train_y, label="train")
        plt.plot(eval_df["step"], eval_y, label="eval")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"{title_prefix}Train vs Validation Loss")
        plt.legend()
        f3 = save_path / "train_vs_eval_loss.png"
        plt.savefig(f3, dpi=200, bbox_inches="tight")
        plt.close()
        files["train_vs_eval_loss"] = str(f3)

    else:
        # Show on screen
        plt.figure()
        plt.plot(train_df["step"], train_y)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"{title_prefix}Training Loss")
        plt.show()

        plt.figure()
        plt.plot(eval_df["step"], eval_y)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"{title_prefix}Validation Loss")
        plt.show()

        plt.figure()
        plt.plot(train_df["step"], train_y, label="train")
        plt.plot(eval_df["step"], eval_y, label="eval")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"{title_prefix}Train vs Validation Loss")
        plt.legend()
        plt.show()

    return {"train_df": train_df, "eval_df": eval_df, "files": files}

if __name__ == "__main__":
    plot_training_eval_curves(
        train_loss_csv="plots/train_loss.csv",
        eval_loss_csv="plots/eval_loss.csv",
        title_prefix="Fold 1 - ",
        smooth_window=None,          # keep raw curves
        save_dir="plots/figures"     # figures saved here
    )
