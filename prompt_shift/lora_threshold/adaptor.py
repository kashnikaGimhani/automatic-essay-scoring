import os
import math
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import cohen_kappa_score

import sys
from pathlib import Path

PARENT_DIR = str(Path(__file__).resolve().parents[1])
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from utils import (
    TRAIT_COLUMNS,
    ensure_dir,
    save_json,
    set_seed,
    normalize_prompt_id,
    parse_prompt_list,
    parse_int_list,
    build_prompt_text_map,
    build_score_ranges_from_hardcoded,
    build_global_trait_fallback,
    AESDataset,
    masked_regression_loss,
    evaluate,
    format_metrics_for_print,
    load_base_checkpoint_into_model,
    apply_lora_to_encoder,
    mark_only_lora_and_head_trainable,
    count_parameters,
    amp_context,
    get_range_for_trait,
    denormalize_score,
    round_to_step,
)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA adaptation on reusable few-shot target splits")

    parser.add_argument("--data_path", type=str, required=True, help="Full original dataset; used for fallback ranges")
    parser.add_argument("--split_root", type=str, required=True, help="Root dir created by create_target_fewshot_splits.py")
    parser.add_argument("--base_root", type=str, required=True, help="Root containing base checkpoints")
    parser.add_argument("--output_root", type=str, required=True)

    parser.add_argument("--sep", type=str, default="\t")
    parser.add_argument("--prompt_col", type=str, default="essay_set")
    parser.add_argument("--text_col", type=str, default="essay")
    parser.add_argument("--id_col", type=str, default="essay_id")

    parser.add_argument("--heldout_prompts", type=str, default="all")
    parser.add_argument("--fewshot_sizes", type=str, default="8,16,32,64,128")

    parser.add_argument("--max_length", type=int, default=480)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dropout_override", type=float, default=-1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "huber"])
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--round_step", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--disable_threshold_tuning", action="store_true",
                        help="Disable dev-set threshold tuning. By default, tuning is applied after each LoRA run.")
    parser.add_argument("--threshold_grid_size", type=int, default=81,
                        help="Number of candidate values in the fixed threshold grid.")
    parser.add_argument("--threshold_max_iters", type=int, default=20,
                        help="Maximum coordinate-search passes for threshold tuning.")

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, default="query,value")
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
    parser.add_argument("--use_rslora", action="store_true", help="Enable rank-stabilized LoRA")
    parser.add_argument("--use_dora", action="store_true", help="Enable DoRA")

    return parser.parse_args()



def predict_to_dataframe(
    model,
    dataloader,
    dataset,
    original_df,
    trait_cols,
    score_ranges,
    global_trait_fallback,
    device,
    round_step,
    id_col,
    prompt_col,
    text_col,
):
    """Save gold, raw continuous predictions, and default rounded predictions.

    Raw predictions are needed for dev-set threshold tuning.
    """
    model.eval()

    all_preds, all_labels, all_masks, all_indices = [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            label_mask = batch["label_mask"].cpu().numpy()
            idxs = batch["idx"].cpu().numpy()
            token_type_ids = batch["token_type_ids"].to(device) if "token_type_ids" in batch else None

            preds = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels)
            all_masks.append(label_mask)
            all_indices.append(idxs)

    if not all_preds:
        return pd.DataFrame()

    preds_norm = np.concatenate(all_preds, axis=0)
    labels_norm = np.concatenate(all_labels, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    indices = np.concatenate(all_indices, axis=0)

    order = np.argsort(indices)
    preds_norm = preds_norm[order]
    labels_norm = labels_norm[order]
    masks = masks[order]
    indices = indices[order]

    rows = []
    df_reset = original_df.reset_index(drop=True)

    for row_pos, ds_idx in enumerate(indices):
        ds_idx = int(ds_idx)
        src_row = df_reset.iloc[ds_idx]
        prompt_id = dataset.prompt_ids[ds_idx]

        row = {
            "row_idx": ds_idx,
            prompt_col: prompt_id,
            text_col: src_row[text_col] if text_col in df_reset.columns else "",
        }
        if id_col in df_reset.columns:
            row[id_col] = src_row[id_col]

        for j, trait in enumerate(trait_cols):
            if masks[row_pos, j] < 0.5:
                row[f"target_{trait}"] = np.nan
                row[f"pred_raw_{trait}"] = np.nan
                row[f"pred_default_{trait}"] = np.nan
                row[f"pred_{trait}"] = np.nan
                continue

            rng = get_range_for_trait(score_ranges, prompt_id, trait, global_trait_fallback)
            mn, mx = float(rng["min"]), float(rng["max"])

            gold_raw = denormalize_score(labels_norm[row_pos, j], mn, mx)
            pred_raw = denormalize_score(preds_norm[row_pos, j], mn, mx)

            gold_rounded = float(np.clip(round_to_step(gold_raw, round_step), mn, mx))
            pred_raw = float(np.clip(pred_raw, mn, mx))
            pred_default = float(np.clip(round_to_step(pred_raw, round_step), mn, mx))

            row[f"target_{trait}"] = gold_rounded
            row[f"pred_raw_{trait}"] = pred_raw
            row[f"pred_default_{trait}"] = pred_default
            row[f"pred_{trait}"] = pred_default  # backward-compatible name

        rows.append(row)

    pred_df = pd.DataFrame(rows)

    base_cols = ["row_idx"]
    if id_col in pred_df.columns:
        base_cols.append(id_col)
    for col in [prompt_col, text_col]:
        if col in pred_df.columns and col not in base_cols:
            base_cols.append(col)

    trait_cols_ordered = []
    for trait in trait_cols:
        for col in [f"target_{trait}", f"pred_raw_{trait}", f"pred_default_{trait}", f"pred_{trait}", f"pred_tuned_{trait}"]:
            if col in pred_df.columns:
                trait_cols_ordered.append(col)

    remaining = [c for c in pred_df.columns if c not in base_cols and c not in trait_cols_ordered]
    return pred_df[base_cols + trait_cols_ordered + remaining]


# ---------------------------------------------------------------------
# Threshold tuning helpers
# ---------------------------------------------------------------------

def valid_score_labels(mn: float, mx: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("round_step must be > 0 for threshold tuning.")
    n = int(round((mx - mn) / step)) + 1
    return np.clip(mn + np.arange(n, dtype=np.float64) * step, mn, mx)


def default_thresholds(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float64)
    if len(labels) <= 1:
        return np.array([], dtype=np.float64)
    return (labels[:-1] + labels[1:]) / 2.0


def scores_to_indices(scores: np.ndarray, labels: np.ndarray, step: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    mn = float(labels[0])
    idx = np.rint((scores - mn) / step).astype(int)
    return np.clip(idx, 0, len(labels) - 1)


def apply_thresholds(pred_raw: np.ndarray, thresholds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    pred_raw = np.asarray(pred_raw, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    idx = np.searchsorted(thresholds, pred_raw, side="right")
    idx = np.clip(idx, 0, len(labels) - 1)
    return labels[idx]


def qwk_from_thresholds(y_true_raw: np.ndarray, pred_raw: np.ndarray, thresholds: np.ndarray, labels: np.ndarray, step: float) -> float:
    if len(y_true_raw) < 2:
        return float("nan")
    y_true_idx = scores_to_indices(y_true_raw, labels, step)
    y_pred_idx = scores_to_indices(apply_thresholds(pred_raw, thresholds, labels), labels, step)
    if len(np.unique(y_true_idx)) < 2:
        return float("nan")
    try:
        return float(cohen_kappa_score(y_true_idx, y_pred_idx, labels=list(range(len(labels))), weights="quadratic"))
    except Exception:
        return float("nan")


def tune_one_trait_thresholds(y_true_raw, pred_raw, labels, step, grid_size=81, max_iters=20):
    y_true_raw = np.asarray(y_true_raw, dtype=np.float64)
    pred_raw = np.asarray(pred_raw, dtype=np.float64)
    valid = ~np.isnan(y_true_raw) & ~np.isnan(pred_raw)
    y_true_raw = y_true_raw[valid]
    pred_raw = pred_raw[valid]

    th = default_thresholds(labels)
    default_qwk = qwk_from_thresholds(y_true_raw, pred_raw, th, labels, step)

    if len(th) == 0 or len(y_true_raw) < 2 or len(np.unique(scores_to_indices(y_true_raw, labels, step))) < 2:
        return {
            "thresholds": [float(x) for x in th.tolist()],
            "labels": [float(x) for x in labels.tolist()],
            "dev_qwk_default": None if math.isnan(default_qwk) else float(default_qwk),
            "dev_qwk_tuned": None if math.isnan(default_qwk) else float(default_qwk),
            "n": int(len(y_true_raw)),
            "used_fallback_default": True,
        }

    mn, mx = float(labels[0]), float(labels[-1])
    candidates = [np.linspace(mn, mx, max(grid_size, 3)), th]
    if len(pred_raw) > 0:
        candidates.append(np.quantile(pred_raw, np.linspace(0.02, 0.98, 49)))
        uniq = np.unique(np.sort(pred_raw))
        if len(uniq) > 1:
            mids = (uniq[:-1] + uniq[1:]) / 2.0
            if len(mids) > 100:
                mids = np.quantile(mids, np.linspace(0.01, 0.99, 100))
            candidates.append(mids)
    candidates = np.unique(np.round(np.clip(np.concatenate(candidates), mn, mx), 8))

    best_th = th.copy()
    best_qwk = default_qwk if not math.isnan(default_qwk) else -1e9
    min_gap = 1e-6

    for _ in range(max_iters):
        improved = False
        for i in range(len(best_th)):
            lower = mn if i == 0 else best_th[i - 1] + min_gap
            upper = mx if i == len(best_th) - 1 else best_th[i + 1] - min_gap
            valid_candidates = candidates[(candidates > lower) & (candidates < upper)]
            for val in valid_candidates:
                trial = best_th.copy()
                trial[i] = float(val)
                qwk = qwk_from_thresholds(y_true_raw, pred_raw, trial, labels, step)
                if not math.isnan(qwk) and qwk > best_qwk + 1e-12:
                    best_qwk = qwk
                    best_th = trial
                    improved = True
        if not improved:
            break

    tuned_qwk = qwk_from_thresholds(y_true_raw, pred_raw, best_th, labels, step)
    return {
        "thresholds": [float(x) for x in best_th.tolist()],
        "labels": [float(x) for x in labels.tolist()],
        "dev_qwk_default": None if math.isnan(default_qwk) else float(default_qwk),
        "dev_qwk_tuned": None if math.isnan(tuned_qwk) else float(tuned_qwk),
        "n": int(len(y_true_raw)),
        "used_fallback_default": False,
    }


def tune_thresholds_by_prompt_trait(dev_pred_df, trait_cols, score_ranges, global_trait_fallback, prompt_col, round_step, grid_size, max_iters):
    threshold_map = {}
    prompts = sorted(dev_pred_df[prompt_col].dropna().astype(str).map(normalize_prompt_id).unique().tolist())
    for prompt_id in prompts:
        prompt_df = dev_pred_df[dev_pred_df[prompt_col].astype(str).map(normalize_prompt_id) == prompt_id]
        threshold_map[prompt_id] = {}
        for trait in trait_cols:
            gold_col = f"target_{trait}"
            raw_col = f"pred_raw_{trait}"
            if gold_col not in prompt_df.columns or raw_col not in prompt_df.columns:
                continue
            valid = prompt_df[[gold_col, raw_col]].dropna()
            if len(valid) == 0:
                continue
            rng = get_range_for_trait(score_ranges, prompt_id, trait, global_trait_fallback)
            mn, mx = float(rng["min"]), float(rng["max"])
            labels = valid_score_labels(mn, mx, round_step)
            result = tune_one_trait_thresholds(
                y_true_raw=valid[gold_col].to_numpy(dtype=np.float64),
                pred_raw=valid[raw_col].to_numpy(dtype=np.float64),
                labels=labels,
                step=round_step,
                grid_size=grid_size,
                max_iters=max_iters,
            )
            result.update({"prompt_id": prompt_id, "trait": trait, "score_min": mn, "score_max": mx, "round_step": round_step})
            threshold_map[prompt_id][trait] = result
    return threshold_map


def apply_threshold_map_to_dataframe(pred_df, threshold_map, trait_cols, score_ranges, global_trait_fallback, prompt_col, round_step):
    out = pred_df.copy()
    for idx, row in out.iterrows():
        prompt_id = normalize_prompt_id(row[prompt_col])
        for trait in trait_cols:
            raw_col = f"pred_raw_{trait}"
            tuned_col = f"pred_tuned_{trait}"
            if raw_col not in out.columns or pd.isna(row.get(raw_col, np.nan)):
                out.at[idx, tuned_col] = np.nan
                continue
            if prompt_id in threshold_map and trait in threshold_map[prompt_id]:
                item = threshold_map[prompt_id][trait]
                labels = np.asarray(item["labels"], dtype=np.float64)
                thresholds = np.asarray(item["thresholds"], dtype=np.float64)
            else:
                rng = get_range_for_trait(score_ranges, prompt_id, trait, global_trait_fallback)
                labels = valid_score_labels(float(rng["min"]), float(rng["max"]), round_step)
                thresholds = default_thresholds(labels)
            out.at[idx, tuned_col] = float(apply_thresholds(np.array([float(row[raw_col])]), thresholds, labels)[0])
    return out


def compute_metrics_from_prediction_df(pred_df, trait_cols, score_ranges, global_trait_fallback, prompt_col, pred_prefix, round_step):
    trait_metrics = {}
    qwk_values, rmse_values = [], []
    for trait in trait_cols:
        gold_col = f"target_{trait}"
        pred_col = f"{pred_prefix}_{trait}"
        if gold_col not in pred_df.columns or pred_col not in pred_df.columns:
            trait_metrics[trait] = {"n": 0, "qwk": float("nan"), "rmse": float("nan")}
            continue
        valid = pred_df[[prompt_col, gold_col, pred_col]].dropna().copy()
        if len(valid) == 0:
            trait_metrics[trait] = {"n": 0, "qwk": float("nan"), "rmse": float("nan")}
            continue

        y_true_all, y_pred_all = [], []
        y_true_raw_all, y_pred_raw_all = [], []
        offset = 0
        for prompt_id, subdf in valid.groupby(valid[prompt_col].astype(str).map(normalize_prompt_id)):
            rng = get_range_for_trait(score_ranges, prompt_id, trait, global_trait_fallback)
            labels = valid_score_labels(float(rng["min"]), float(rng["max"]), round_step)
            yt = subdf[gold_col].to_numpy(dtype=np.float64)
            yp = subdf[pred_col].to_numpy(dtype=np.float64)
            y_true_all.extend((scores_to_indices(yt, labels, round_step) + offset).tolist())
            y_pred_all.extend((scores_to_indices(yp, labels, round_step) + offset).tolist())
            offset += len(labels)
            y_true_raw_all.extend(yt.tolist())
            y_pred_raw_all.extend(yp.tolist())

        y_true_all = np.asarray(y_true_all, dtype=int)
        y_pred_all = np.asarray(y_pred_all, dtype=int)
        if len(y_true_all) < 2 or len(np.unique(y_true_all)) < 2:
            qwk = float("nan")
        else:
            try:
                qwk = float(cohen_kappa_score(y_true_all, y_pred_all, weights="quadratic"))
            except Exception:
                qwk = float("nan")
        rmse = float(np.sqrt(np.mean((np.asarray(y_true_raw_all) - np.asarray(y_pred_raw_all)) ** 2)))
        if not math.isnan(qwk):
            qwk_values.append(qwk)
        if not math.isnan(rmse):
            rmse_values.append(rmse)
        trait_metrics[trait] = {"n": int(len(valid)), "qwk": qwk, "rmse": rmse}

    return {
        "loss": float("nan"),
        "mean_qwk": float(np.mean(qwk_values)) if qwk_values else float("nan"),
        "mean_rmse": float(np.mean(rmse_values)) if rmse_values else float("nan"),
        "trait_metrics": trait_metrics,
    }


def add_threshold_trait_rows(rows, heldout_prompt, repeat_name, fewshot_k, model_type, trait_cols, dev_default, dev_tuned, test_default, test_tuned):
    for trait in trait_cols:
        dd = dev_default.get("trait_metrics", {}).get(trait, {})
        dt = dev_tuned.get("trait_metrics", {}).get(trait, {})
        td = test_default.get("trait_metrics", {}).get(trait, {})
        tt = test_tuned.get("trait_metrics", {}).get(trait, {})
        rows.append({
            "heldout_prompt": heldout_prompt,
            "repeat_name": repeat_name,
            "fewshot_k": fewshot_k,
            "model_type": model_type,
            "trait": trait,
            "dev_default_qwk": dd.get("qwk", float("nan")),
            "dev_tuned_qwk": dt.get("qwk", float("nan")),
            "test_default_qwk": td.get("qwk", float("nan")),
            "test_tuned_qwk": tt.get("qwk", float("nan")),
            "test_n": tt.get("n", td.get("n", 0)),
        })


def save_final_trait_mean_qwk_report(trait_rows, output_root, trait_cols):
    if not trait_rows:
        return
    df = pd.DataFrame(trait_rows)
    by_repeat_path = os.path.join(output_root, "lora_threshold_trait_qwk_by_repeat.csv")
    df.to_csv(by_repeat_path, index=False)

    long_rows = []
    for (heldout_prompt, fewshot_k, model_type, trait), g in df.groupby(["heldout_prompt", "fewshot_k", "model_type", "trait"], dropna=False):
        tuned_vals = pd.to_numeric(g["test_tuned_qwk"], errors="coerce").dropna()
        default_vals = pd.to_numeric(g["test_default_qwk"], errors="coerce").dropna()
        long_rows.append({
            "heldout_prompt": heldout_prompt,
            "fewshot_k": fewshot_k,
            "model_type": model_type,
            "trait": trait,
            "n_repeats": int(g["repeat_name"].nunique()),
            "test_default_qwk_mean": float(default_vals.mean()) if len(default_vals) else float("nan"),
            "test_tuned_qwk_mean": float(tuned_vals.mean()) if len(tuned_vals) else float("nan"),
            "test_tuned_qwk_std": float(tuned_vals.std(ddof=0)) if len(tuned_vals) else float("nan"),
        })
    pd.DataFrame(long_rows).to_csv(os.path.join(output_root, "lora_threshold_trait_mean_qwk_across_repeats.csv"), index=False)

    final_rows = []
    for (heldout_prompt, fewshot_k, model_type), g in df.groupby(["heldout_prompt", "fewshot_k", "model_type"], dropna=False):
        row = {
            "heldout_prompt": heldout_prompt,
            "fewshot_k": fewshot_k,
            "model_type": model_type,
            "n_repeats": int(g["repeat_name"].nunique()),
        }
        trait_means = []
        for trait in trait_cols:
            tuned_vals = pd.to_numeric(g.loc[g["trait"] == trait, "test_tuned_qwk"], errors="coerce").dropna()
            default_vals = pd.to_numeric(g.loc[g["trait"] == trait, "test_default_qwk"], errors="coerce").dropna()
            tuned_mean = float(tuned_vals.mean()) if len(tuned_vals) else float("nan")
            default_mean = float(default_vals.mean()) if len(default_vals) else float("nan")
            row[f"{trait}_mean_qwk"] = tuned_mean
            row[f"{trait}_default_mean_qwk"] = default_mean
            if not math.isnan(tuned_mean):
                trait_means.append(tuned_mean)
        row["mean_qwk_across_available_traits"] = float(np.mean(trait_means)) if trait_means else float("nan")
        final_rows.append(row)

    final_df = pd.DataFrame(final_rows)
    ordered = ["heldout_prompt", "fewshot_k", "model_type", "n_repeats"]
    for trait in trait_cols:
        ordered += [f"{trait}_mean_qwk", f"{trait}_default_mean_qwk"]
    ordered.append("mean_qwk_across_available_traits")
    ordered = [c for c in ordered if c in final_df.columns]
    final_df = final_df[ordered + [c for c in final_df.columns if c not in ordered]]
    final_df.to_csv(os.path.join(output_root, "final_trait_mean_qwk_report.csv"), index=False)
    print(f"Saved final trait mean QWK report: {os.path.join(output_root, 'final_trait_mean_qwk_report.csv')}", flush=True)


def flatten_trait_metrics(prefix, trait_metrics, trait_cols):
    row = {}
    for trait in trait_cols:
        tm = trait_metrics.get(trait, {})
        row[f"{prefix}_{trait}_n"] = tm.get("n", 0)
        row[f"{prefix}_{trait}_qwk"] = tm.get("qwk", float("nan"))
        row[f"{prefix}_{trait}_rmse"] = tm.get("rmse", float("nan"))
    return row



def build_mean_trait_qwk_across_repeats(df, trait_cols, score_column_prefix="test"):
    if df.empty:
        return pd.DataFrame(columns=["heldout_prompt", "fewshot_k", "trait", "mean_qwk", "std_qwk", "num_repeats"])

    rows = []
    grouped = df.groupby(["heldout_prompt", "fewshot_k"], dropna=False)
    for (heldout_prompt, fewshot_k), g in grouped:
        for trait in trait_cols:
            col = f"{score_column_prefix}_{trait}_qwk"
            if col not in g.columns:
                continue
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            rows.append(
                {
                    "heldout_prompt": heldout_prompt,
                    "fewshot_k": fewshot_k,
                    "trait": trait,
                    "mean_qwk": float(vals.mean()) if len(vals) else float("nan"),
                    "std_qwk": float(vals.std(ddof=0)) if len(vals) else float("nan"),
                    "num_repeats": int(len(vals)),
                }
            )
    return pd.DataFrame(rows)



def train_one_run(
    model,
    train_loader,
    dev_loader,
    dev_dataset,
    trait_cols,
    score_ranges,
    global_trait_fallback,
    device,
    args,
    run_dir,
):
    mark_only_lora_and_head_trainable(model)
    param_counts = count_parameters(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    total_update_steps = max(1, math.ceil(len(train_loader) / args.grad_accum_steps) * args.num_epochs)
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    amp_enabled, autocast_device, autocast_dtype = amp_context(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_dev_qwk = -1e9
    best_epoch = -1
    early_stop_counter = 0
    best_ckpt_path = os.path.join(run_dir, "best_lora_model.pt")
    history = []

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        num_steps = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}", leave=False)

        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            label_mask = batch["label_mask"].to(device)

            with torch.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=amp_enabled):
                preds = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = masked_regression_loss(
                    preds=preds,
                    targets=labels,
                    mask=label_mask,
                    loss_type=args.loss_type,
                    huber_delta=args.huber_delta,
                )
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running_loss += loss.item() * args.grad_accum_steps
            num_steps += 1
            progress.set_postfix(loss=f"{running_loss / max(num_steps, 1):.4f}")

        train_loss = running_loss / max(num_steps, 1)
        dev_metrics = evaluate(
            model=model,
            dataloader=dev_loader,
            dataset=dev_dataset,
            trait_cols=trait_cols,
            score_ranges=score_ranges,
            global_trait_fallback=global_trait_fallback,
            device=device,
            round_step=args.round_step,
            loss_type=args.loss_type,
            huber_delta=args.huber_delta,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_metrics["loss"],
            "dev_mean_qwk": dev_metrics["mean_qwk"],
            "dev_mean_rmse": dev_metrics["mean_rmse"],
        })

        dev_qwk = dev_metrics["mean_qwk"]
        improved = not math.isnan(dev_qwk) and dev_qwk > best_dev_qwk

        if improved:
            best_dev_qwk = dev_qwk
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_dev_mean_qwk": best_dev_qwk,
                    "trait_cols": trait_cols,
                    "param_counts": param_counts,
                    "lora_config": {
                        "r": args.lora_r,
                        "alpha": args.lora_alpha,
                        "dropout": args.lora_dropout,
                        "target_modules": [x.strip() for x in args.lora_target_modules.split(",") if x.strip()],
                        "bias": args.lora_bias,
                        "use_rslora": args.use_rslora,
                        "use_dora": args.use_dora,
                    },
                },
                best_ckpt_path,
            )
        else:
            early_stop_counter += 1

        if early_stop_counter >= args.patience:
            break

    if os.path.exists(best_ckpt_path):
        best_state = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_state["model_state_dict"], strict=True)

    return model, history, best_epoch, best_dev_qwk, param_counts



def main():
    args = parse_args()
    ensure_dir(args.output_root)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_df = pd.read_csv(args.data_path, sep=args.sep)
    full_df[args.prompt_col] = full_df[args.prompt_col].apply(normalize_prompt_id)
    for trait in TRAIT_COLUMNS:
        if trait in full_df.columns:
            full_df[trait] = pd.to_numeric(full_df[trait], errors="coerce")

    all_prompts = sorted(full_df[args.prompt_col].astype(str).unique().tolist())
    heldout_prompts = parse_prompt_list(args.heldout_prompts, all_prompts)
    fewshot_sizes = sorted(parse_int_list(args.fewshot_sizes))
    lora_target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]

    prompt_text_map = build_prompt_text_map()
    score_ranges = build_score_ranges_from_hardcoded()
    global_trait_fallback = build_global_trait_fallback(full_df, TRAIT_COLUMNS)

    summary_rows = []
    repeat_test_rows = []
    threshold_trait_rows = []

    for heldout_prompt in heldout_prompts:
        split_prompt_dir = os.path.join(args.split_root, f"heldout_{heldout_prompt}")
        base_ckpt_dir = os.path.join(args.base_root, f"base_prompt{heldout_prompt}", "best_checkpoint")

        if not os.path.isdir(split_prompt_dir):
            print(f"Skipping heldout={heldout_prompt}: split dir not found -> {split_prompt_dir}")
            continue
        if not os.path.isdir(base_ckpt_dir):
            print(f"Skipping heldout={heldout_prompt}: base checkpoint dir not found -> {base_ckpt_dir}")
            continue

        repeat_dirs = sorted(
            [
                os.path.join(split_prompt_dir, d)
                for d in os.listdir(split_prompt_dir)
                if d.startswith("repeat_") and os.path.isdir(os.path.join(split_prompt_dir, d))
            ]
        )

        for repeat_dir in repeat_dirs:
            repeat_name = os.path.basename(repeat_dir)
            print(f"\n=== heldout={heldout_prompt} | {repeat_name} ===")

            dev_df = pd.read_csv(os.path.join(repeat_dir, "dev.tsv"), sep="\t")
            test_df = pd.read_csv(os.path.join(repeat_dir, "test.tsv"), sep="\t")

            tokenizer = AutoTokenizer.from_pretrained(base_ckpt_dir, use_fast=True)
            dev_dataset = AESDataset(
                df=dev_df,
                tokenizer=tokenizer,
                trait_cols=TRAIT_COLUMNS,
                prompt_col=args.prompt_col,
                text_col=args.text_col,
                prompt_text_map=prompt_text_map,
                score_ranges=score_ranges,
                global_trait_fallback=global_trait_fallback,
                max_length=args.max_length,
            )
            test_dataset = AESDataset(
                df=test_df,
                tokenizer=tokenizer,
                trait_cols=TRAIT_COLUMNS,
                prompt_col=args.prompt_col,
                text_col=args.text_col,
                prompt_text_map=prompt_text_map,
                score_ranges=score_ranges,
                global_trait_fallback=global_trait_fallback,
                max_length=args.max_length,
            )
            dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

            repeat_out_dir = os.path.join(args.output_root, f"heldout_{heldout_prompt}", repeat_name)
            ensure_dir(repeat_out_dir)

            for k in fewshot_sizes:
                train_path = os.path.join(repeat_dir, f"fewshot_{k}.tsv")
                if not os.path.exists(train_path):
                    print(f"Skipping k={k}: split not found -> {train_path}")
                    continue

                train_df = pd.read_csv(train_path, sep="\t")
                if len(train_df) == 0:
                    print(f"Skipping k={k}: empty train split")
                    continue

                train_dataset = AESDataset(
                    df=train_df,
                    tokenizer=tokenizer,
                    trait_cols=TRAIT_COLUMNS,
                    prompt_col=args.prompt_col,
                    text_col=args.text_col,
                    prompt_text_map=prompt_text_map,
                    score_ranges=score_ranges,
                    global_trait_fallback=global_trait_fallback,
                    max_length=args.max_length,
                )
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

                model, trait_cols, _ = load_base_checkpoint_into_model(
                    base_ckpt_dir=base_ckpt_dir,
                    device=device,
                    dropout_override=args.dropout_override,
                )
                model = apply_lora_to_encoder(
                    model=model,
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=lora_target_modules,
                    bias=args.lora_bias,
                    use_rslora=args.use_rslora,
                    use_dora=args.use_dora,
                )
                model.to(device)

                run_dir = os.path.join(repeat_out_dir, f"k_{k}")
                ensure_dir(run_dir)

                model, history, best_epoch, best_dev_qwk, param_counts = train_one_run(
                    model=model,
                    train_loader=train_loader,
                    dev_loader=dev_loader,
                    dev_dataset=dev_dataset,
                    trait_cols=trait_cols,
                    score_ranges=score_ranges,
                    global_trait_fallback=global_trait_fallback,
                    device=device,
                    args=args,
                    run_dir=run_dir,
                )

                final_dev = evaluate(
                    model=model,
                    dataloader=dev_loader,
                    dataset=dev_dataset,
                    trait_cols=trait_cols,
                    score_ranges=score_ranges,
                    global_trait_fallback=global_trait_fallback,
                    device=device,
                    round_step=args.round_step,
                    loss_type=args.loss_type,
                    huber_delta=args.huber_delta,
                )
                final_test = evaluate(
                    model=model,
                    dataloader=test_loader,
                    dataset=test_dataset,
                    trait_cols=trait_cols,
                    score_ranges=score_ranges,
                    global_trait_fallback=global_trait_fallback,
                    device=device,
                    round_step=args.round_step,
                    loss_type=args.loss_type,
                    huber_delta=args.huber_delta,
                )

                final_dev_predictions = predict_to_dataframe(
                    model=model,
                    dataloader=dev_loader,
                    dataset=dev_dataset,
                    original_df=dev_df,
                    trait_cols=trait_cols,
                    score_ranges=score_ranges,
                    global_trait_fallback=global_trait_fallback,
                    device=device,
                    round_step=args.round_step,
                    id_col=args.id_col,
                    prompt_col=args.prompt_col,
                    text_col=args.text_col,
                )
                final_test_predictions = predict_to_dataframe(
                    model=model,
                    dataloader=test_loader,
                    dataset=test_dataset,
                    original_df=test_df,
                    trait_cols=trait_cols,
                    score_ranges=score_ranges,
                    global_trait_fallback=global_trait_fallback,
                    device=device,
                    round_step=args.round_step,
                    id_col=args.id_col,
                    prompt_col=args.prompt_col,
                    text_col=args.text_col,
                )

                threshold_map = None
                tuned_dev_metrics = None
                tuned_test_metrics = None
                tuned_dev_predictions = None
                tuned_test_predictions = None

                if not args.disable_threshold_tuning:
                    threshold_map = tune_thresholds_by_prompt_trait(
                        dev_pred_df=final_dev_predictions,
                        trait_cols=trait_cols,
                        score_ranges=score_ranges,
                        global_trait_fallback=global_trait_fallback,
                        prompt_col=args.prompt_col,
                        round_step=args.round_step,
                        grid_size=args.threshold_grid_size,
                        max_iters=args.threshold_max_iters,
                    )
                    tuned_dev_predictions = apply_threshold_map_to_dataframe(
                        final_dev_predictions, threshold_map, trait_cols,
                        score_ranges, global_trait_fallback, args.prompt_col, args.round_step,
                    )
                    tuned_test_predictions = apply_threshold_map_to_dataframe(
                        final_test_predictions, threshold_map, trait_cols,
                        score_ranges, global_trait_fallback, args.prompt_col, args.round_step,
                    )
                    tuned_dev_metrics = compute_metrics_from_prediction_df(
                        tuned_dev_predictions, trait_cols, score_ranges,
                        global_trait_fallback, args.prompt_col, "pred_tuned", args.round_step,
                    )
                    tuned_test_metrics = compute_metrics_from_prediction_df(
                        tuned_test_predictions, trait_cols, score_ranges,
                        global_trait_fallback, args.prompt_col, "pred_tuned", args.round_step,
                    )
                    add_threshold_trait_rows(
                        threshold_trait_rows, heldout_prompt, repeat_name, k, "lora",
                        trait_cols, final_dev, tuned_dev_metrics, final_test, tuned_test_metrics,
                    )

                save_json({"history": history}, os.path.join(run_dir, "training_history.json"))
                save_json(final_dev, os.path.join(run_dir, "final_dev_metrics.json"))
                save_json(final_test, os.path.join(run_dir, "final_test_metrics.json"))
                final_dev_predictions.to_csv(os.path.join(run_dir, "final_dev_predictions.csv"), index=False)
                final_test_predictions.to_csv(os.path.join(run_dir, "final_test_predictions.csv"), index=False)
                if threshold_map is not None:
                    save_json(threshold_map, os.path.join(run_dir, "thresholds_by_prompt_trait.json"))
                    save_json(tuned_dev_metrics, os.path.join(run_dir, "final_dev_threshold_tuned_metrics.json"))
                    save_json(tuned_test_metrics, os.path.join(run_dir, "final_test_threshold_tuned_metrics.json"))
                    tuned_dev_predictions.to_csv(os.path.join(run_dir, "final_dev_predictions_thresholded.csv"), index=False)
                    tuned_test_predictions.to_csv(os.path.join(run_dir, "final_test_predictions_thresholded.csv"), index=False)
                save_json(
                    {
                        "heldout_prompt": heldout_prompt,
                        "repeat_name": repeat_name,
                        "fewshot_k": k,
                        "best_epoch": best_epoch,
                        "best_dev_mean_qwk": best_dev_qwk,
                        "base_checkpoint_dir": base_ckpt_dir,
                        "lora_r": args.lora_r,
                        "lora_alpha": args.lora_alpha,
                        "lora_dropout": args.lora_dropout,
                        "lora_target_modules": lora_target_modules,
                        "lora_bias": args.lora_bias,
                        "param_counts": param_counts,
                        "train_n": len(train_df),
                        "dev_n": len(dev_df),
                        "test_n": len(test_df),
                        "use_dora": args.use_dora,
                        "use_rslora": args.use_rslora,
                    },
                    os.path.join(run_dir, "run_config.json"),
                )

                print(f"\nLoRA | heldout={heldout_prompt} | {repeat_name} | k={k}")
                print(f"Trainable params: {param_counts['trainable']:,} / {param_counts['total']:,}")
                print(format_metrics_for_print("Final dev", final_dev))
                print(format_metrics_for_print("Final test", final_test))
                if tuned_dev_metrics is not None and tuned_test_metrics is not None:
                    print(format_metrics_for_print("Threshold-tuned dev", tuned_dev_metrics))
                    print(format_metrics_for_print("Threshold-tuned test", tuned_test_metrics))

                base_row = {
                    "heldout_prompt": heldout_prompt,
                    "repeat_name": repeat_name,
                    "fewshot_k": k,
                    "train_n": len(train_df),
                    "dev_n": len(dev_df),
                    "test_n": len(test_df),
                    "best_epoch": best_epoch,
                    "best_dev_mean_qwk": best_dev_qwk,
                    "final_dev_mean_qwk": final_dev["mean_qwk"],
                    "final_dev_mean_rmse": final_dev["mean_rmse"],
                    "final_test_mean_qwk": final_test["mean_qwk"],
                    "final_test_mean_rmse": final_test["mean_rmse"],
                    "trainable_params": param_counts["trainable"],
                    "total_params": param_counts["total"],
                }
                base_row.update(flatten_trait_metrics("dev", final_dev["trait_metrics"], trait_cols))
                base_row.update(flatten_trait_metrics("test", final_test["trait_metrics"], trait_cols))
                if tuned_dev_metrics is not None and tuned_test_metrics is not None:
                    base_row["threshold_dev_mean_qwk"] = tuned_dev_metrics["mean_qwk"]
                    base_row["threshold_dev_mean_rmse"] = tuned_dev_metrics["mean_rmse"]
                    base_row["threshold_test_mean_qwk"] = tuned_test_metrics["mean_qwk"]
                    base_row["threshold_test_mean_rmse"] = tuned_test_metrics["mean_rmse"]
                    base_row.update(flatten_trait_metrics("dev_tuned", tuned_dev_metrics["trait_metrics"], trait_cols))
                    base_row.update(flatten_trait_metrics("test_tuned", tuned_test_metrics["trait_metrics"], trait_cols))

                summary_rows.append(base_row)
                repeat_test_rows.append(base_row.copy())

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(os.path.join(args.output_root, "lora_summary.csv"), index=False)

    if repeat_test_rows:
        repeat_test_df = pd.DataFrame(repeat_test_rows)
        repeat_test_df.to_csv(os.path.join(args.output_root, "lora_repeat_test_results.csv"), index=False)

        mean_trait_qwk_df = build_mean_trait_qwk_across_repeats(
            repeat_test_df,
            trait_cols=TRAIT_COLUMNS,
            score_column_prefix="test",
        )
        mean_trait_qwk_df.to_csv(
            os.path.join(args.output_root, "lora_mean_trait_qwk_across_repeats.csv"),
            index=False,
        )

    if threshold_trait_rows:
        save_final_trait_mean_qwk_report(threshold_trait_rows, args.output_root, TRAIT_COLUMNS)

    print("\nDone.")


if __name__ == "__main__":
    main()
