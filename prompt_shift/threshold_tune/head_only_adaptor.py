#!/usr/bin/env python3
"""
Threshold tuning for AES base / head-only adaptor outputs.

What this script does
---------------------
1. Loads the base checkpoint for each held-out prompt.
2. Optionally loads each trained head-only checkpoint: best_head_only.pt.
3. Runs the model on dev/test splits and collects continuous raw predictions.
4. Tunes monotonic thresholds on the DEV split, separately for each prompt and trait.
5. Applies the learned thresholds to the TEST split.
6. Saves baseline rounded metrics, threshold-tuned metrics, prediction CSVs, thresholds JSON,
   one run-level summary CSV, trait-wise aggregate CSVs, and a final compact trait mean QWK report.

Expected existing structure
---------------------------
split_root/
  heldout_2/
    repeat_01/
      dev.tsv
      test.tsv
      fewshot_32.tsv
      ...

base_root/
  base_prompt2/
    best_checkpoint/
      best_model.pt
      tokenizer files...

head_only_root/
  heldout_2/
    repeat_01/
      k_32/
        best_head_only.pt
      k_64/
        best_head_only.pt

Place this file next to utils.py, then run it from the project root.
"""

import os
import math
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import cohen_kappa_score, mean_squared_error

import sys
from pathlib import Path

PARENT_DIR = str(Path(__file__).resolve().parents[1])
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from utils import (
    TRAIT_COLUMNS,
    AESDataset,
    build_global_trait_fallback,
    build_prompt_text_map,
    build_score_ranges_from_hardcoded,
    denormalize_score,
    ensure_dir,
    get_range_for_trait,
    load_base_checkpoint_into_model,
    normalize_prompt_id,
    parse_int_list,
    parse_prompt_list,
    round_to_step,
    save_json,
    set_seed,
)


# -----------------------------
# Argument parsing
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune per-prompt/per-trait thresholds for AES base/head-only models."
    )

    parser.add_argument("--data_path", type=str, required=True,
                        help="Full original dataset; used for global fallback score ranges.")
    parser.add_argument("--split_root", type=str, required=True,
                        help="Root directory containing heldout_{prompt}/repeat_xx/dev.tsv and test.tsv.")
    parser.add_argument("--base_root", type=str, required=True,
                        help="Root containing base checkpoints, e.g. outputs/base_prompt2/best_checkpoint.")
    parser.add_argument("--head_only_root", type=str, required=True,
                        help="Root containing trained head-only runs, e.g. head_only_adaptor/heldout_2/repeat_01/k_32.")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Where threshold tuning outputs will be written.")

    parser.add_argument("--sep", type=str, default="\t")
    parser.add_argument("--prompt_col", type=str, default="essay_set")
    parser.add_argument("--text_col", type=str, default="essay")
    parser.add_argument("--id_col", type=str, default="essay_id")

    parser.add_argument("--heldout_prompts", type=str, default="all",
                        help="Comma-separated prompts, e.g. 2,3,4 or all.")
    parser.add_argument("--fewshot_sizes", type=str, default="8,16,32,64,128")

    parser.add_argument("--max_length", type=int, default=480)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dropout_override", type=float, default=-1.0)
    parser.add_argument("--round_step", type=float, default=1.0,
                        help="Valid score interval. Use 1.0 for integer scores, 0.5 for half-point scales.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--include_zero_shot", action="store_true",
                        help="Also tune/apply thresholds for the base zero-shot model.")
    parser.add_argument("--grid_size", type=int, default=81,
                        help="Number of candidate threshold values in the fixed grid.")
    parser.add_argument("--max_coord_iters", type=int, default=30,
                        help="Coordinate-search iterations per restart.")
    parser.add_argument("--n_random_restarts", type=int, default=20,
                        help="Random/jittered restarts for threshold search.")

    return parser.parse_args()


# -----------------------------
# Score labels and threshold helpers
# -----------------------------

def valid_score_labels(mn: float, mx: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("round_step must be > 0 for threshold tuning.")
    n = int(round((mx - mn) / step)) + 1
    labels = mn + np.arange(n, dtype=np.float64) * step
    labels = np.clip(labels, mn, mx)
    return labels


def default_midpoint_thresholds(labels: np.ndarray) -> np.ndarray:
    if len(labels) <= 1:
        return np.array([], dtype=np.float64)
    return (labels[:-1] + labels[1:]) / 2.0


def scores_to_indices(scores: np.ndarray, labels: np.ndarray, step: float) -> np.ndarray:
    mn = float(labels[0])
    idx = np.rint((scores - mn) / step).astype(int)
    return np.clip(idx, 0, len(labels) - 1)


def apply_thresholds_to_raw_predictions(
    pred_raw: np.ndarray,
    thresholds: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    pred_raw = np.asarray(pred_raw, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    class_idx = np.searchsorted(thresholds, pred_raw, side="right")
    class_idx = np.clip(class_idx, 0, len(labels) - 1)
    return labels[class_idx]


def qwk_for_thresholds(
    y_true_raw: np.ndarray,
    pred_raw: np.ndarray,
    thresholds: np.ndarray,
    labels: np.ndarray,
    step: float,
) -> float:
    if len(y_true_raw) < 2:
        return float("nan")

    y_true_idx = scores_to_indices(y_true_raw, labels, step)
    y_pred_raw = apply_thresholds_to_raw_predictions(pred_raw, thresholds, labels)
    y_pred_idx = scores_to_indices(y_pred_raw, labels, step)

    if len(np.unique(y_true_idx)) < 2:
        return float("nan")

    try:
        return float(cohen_kappa_score(
            y_true_idx,
            y_pred_idx,
            labels=list(range(len(labels))),
            weights="quadratic",
        ))
    except Exception:
        return float("nan")


def safe_qwk_raw(y_true_raw: np.ndarray, y_pred_raw: np.ndarray, labels: np.ndarray, step: float) -> float:
    if len(y_true_raw) < 2:
        return float("nan")

    y_true_idx = scores_to_indices(y_true_raw, labels, step)
    y_pred_idx = scores_to_indices(y_pred_raw, labels, step)

    if len(np.unique(y_true_idx)) < 2:
        return float("nan")

    try:
        return float(cohen_kappa_score(
            y_true_idx,
            y_pred_idx,
            labels=list(range(len(labels))),
            weights="quadratic",
        ))
    except Exception:
        return float("nan")


# -----------------------------
# Threshold optimization
# -----------------------------

def make_candidate_values(
    pred_raw: np.ndarray,
    labels: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    mn, mx = float(labels[0]), float(labels[-1])
    default_th = default_midpoint_thresholds(labels)

    fixed_grid = np.linspace(mn, mx, max(grid_size, 3))

    pred_raw = np.asarray(pred_raw, dtype=np.float64)
    pred_raw = pred_raw[~np.isnan(pred_raw)]

    pieces = [fixed_grid, default_th]

    if len(pred_raw) > 0:
        quantiles = np.quantile(pred_raw, np.linspace(0.02, 0.98, 49))
        pieces.append(quantiles)

        uniq = np.unique(np.sort(pred_raw))
        if len(uniq) > 1:
            mids = (uniq[:-1] + uniq[1:]) / 2.0
            if len(mids) > 100:
                mids = np.quantile(mids, np.linspace(0.01, 0.99, 100))
            pieces.append(mids)

    candidates = np.concatenate(pieces)
    candidates = np.clip(candidates, mn, mx)
    candidates = np.unique(np.round(candidates, 8))
    return candidates


def distribution_matching_init(
    y_true_raw: np.ndarray,
    pred_raw: np.ndarray,
    labels: np.ndarray,
    step: float,
) -> np.ndarray:
    """Initialize thresholds so predicted label distribution roughly matches dev label distribution."""
    if len(labels) <= 1:
        return np.array([], dtype=np.float64)

    y_idx = scores_to_indices(y_true_raw, labels, step)
    counts = np.bincount(y_idx, minlength=len(labels)).astype(np.float64)
    proportions = counts / max(counts.sum(), 1.0)
    cumulative = np.cumsum(proportions)[:-1]

    pred_raw = np.asarray(pred_raw, dtype=np.float64)
    if len(pred_raw) == 0 or np.all(np.isnan(pred_raw)):
        return default_midpoint_thresholds(labels)

    cumulative = np.clip(cumulative, 0.0, 1.0)
    th = np.quantile(pred_raw, cumulative)
    return np.sort(th.astype(np.float64))


def enforce_monotonic_thresholds(th: np.ndarray, labels: np.ndarray, min_gap: float = 1e-6) -> np.ndarray:
    if len(th) == 0:
        return th
    th = np.asarray(th, dtype=np.float64).copy()
    mn, mx = float(labels[0]), float(labels[-1])
    th = np.clip(th, mn, mx)
    th.sort()

    for i in range(1, len(th)):
        if th[i] <= th[i - 1]:
            th[i] = th[i - 1] + min_gap

    if th[-1] > mx:
        th[-1] = mx
        for i in range(len(th) - 2, -1, -1):
            if th[i] >= th[i + 1]:
                th[i] = th[i + 1] - min_gap

    return np.clip(th, mn, mx)


def coordinate_search_thresholds(
    y_true_raw: np.ndarray,
    pred_raw: np.ndarray,
    labels: np.ndarray,
    step: float,
    init_thresholds: np.ndarray,
    candidates: np.ndarray,
    max_iters: int,
) -> Tuple[np.ndarray, float]:
    th = enforce_monotonic_thresholds(init_thresholds, labels)
    best_qwk = qwk_for_thresholds(y_true_raw, pred_raw, th, labels, step)
    if math.isnan(best_qwk):
        best_qwk = -1e9

    if len(th) == 0:
        return th, best_qwk

    min_gap = 1e-6

    for _ in range(max_iters):
        improved_any = False

        for i in range(len(th)):
            lower = float(labels[0]) if i == 0 else th[i - 1] + min_gap
            upper = float(labels[-1]) if i == len(th) - 1 else th[i + 1] - min_gap

            valid_candidates = candidates[(candidates > lower) & (candidates < upper)]
            if len(valid_candidates) == 0:
                continue

            local_best_val = th[i]
            local_best_qwk = best_qwk

            for val in valid_candidates:
                trial = th.copy()
                trial[i] = float(val)
                trial_qwk = qwk_for_thresholds(y_true_raw, pred_raw, trial, labels, step)
                if math.isnan(trial_qwk):
                    continue
                if trial_qwk > local_best_qwk + 1e-12:
                    local_best_qwk = trial_qwk
                    local_best_val = float(val)

            if local_best_qwk > best_qwk + 1e-12:
                th[i] = local_best_val
                best_qwk = local_best_qwk
                improved_any = True

        if not improved_any:
            break

    return enforce_monotonic_thresholds(th, labels), float(best_qwk)


def tune_thresholds_for_one_trait(
    y_true_raw: np.ndarray,
    pred_raw: np.ndarray,
    labels: np.ndarray,
    step: float,
    seed: int,
    grid_size: int,
    max_coord_iters: int,
    n_random_restarts: int,
) -> Dict[str, Any]:
    y_true_raw = np.asarray(y_true_raw, dtype=np.float64)
    pred_raw = np.asarray(pred_raw, dtype=np.float64)

    valid = ~np.isnan(y_true_raw) & ~np.isnan(pred_raw)
    y_true_raw = y_true_raw[valid]
    pred_raw = pred_raw[valid]

    default_th = default_midpoint_thresholds(labels)
    default_qwk = qwk_for_thresholds(y_true_raw, pred_raw, default_th, labels, step)

    if len(y_true_raw) < 2 or len(np.unique(scores_to_indices(y_true_raw, labels, step))) < 2:
        return {
            "thresholds": default_th.tolist(),
            "labels": labels.tolist(),
            "dev_qwk_default": default_qwk,
            "dev_qwk_tuned": default_qwk,
            "n": int(len(y_true_raw)),
            "used_fallback_default": True,
            "reason": "Too few samples or only one gold class in dev split.",
        }

    candidates = make_candidate_values(pred_raw, labels, grid_size)
    rng = np.random.RandomState(seed)

    initializations = []
    initializations.append(default_th)
    initializations.append(distribution_matching_init(y_true_raw, pred_raw, labels, step))

    if len(labels) > 1 and len(pred_raw) > 0:
        uniform_quantiles = np.linspace(1.0 / len(labels), (len(labels) - 1.0) / len(labels), len(labels) - 1)
        initializations.append(np.quantile(pred_raw, uniform_quantiles))

    for _ in range(n_random_restarts):
        base = initializations[rng.randint(0, len(initializations))]
        if len(base) == 0:
            jittered = base
        else:
            scale = max((float(labels[-1]) - float(labels[0])) * 0.05, step * 0.1)
            jittered = base + rng.normal(loc=0.0, scale=scale, size=len(base))
        initializations.append(jittered)

    best_th = default_th
    best_qwk = default_qwk
    if math.isnan(best_qwk):
        best_qwk = -1e9

    for init in initializations:
        trial_th, trial_qwk = coordinate_search_thresholds(
            y_true_raw=y_true_raw,
            pred_raw=pred_raw,
            labels=labels,
            step=step,
            init_thresholds=init,
            candidates=candidates,
            max_iters=max_coord_iters,
        )
        if not math.isnan(trial_qwk) and trial_qwk > best_qwk + 1e-12:
            best_qwk = trial_qwk
            best_th = trial_th

    tuned_qwk = qwk_for_thresholds(y_true_raw, pred_raw, best_th, labels, step)

    return {
        "thresholds": [float(x) for x in best_th.tolist()],
        "labels": [float(x) for x in labels.tolist()],
        "dev_qwk_default": None if math.isnan(default_qwk) else float(default_qwk),
        "dev_qwk_tuned": None if math.isnan(tuned_qwk) else float(tuned_qwk),
        "n": int(len(y_true_raw)),
        "used_fallback_default": False,
        "reason": "optimized_on_dev_qwk",
    }


# -----------------------------
# Prediction extraction
# -----------------------------

def load_head_only_checkpoint_into_base_model(
    base_ckpt_dir: str,
    head_only_ckpt_path: str,
    device: torch.device,
    dropout_override: float,
):
    model, trait_cols, base_meta = load_base_checkpoint_into_model(
        base_ckpt_dir=base_ckpt_dir,
        device=device,
        dropout_override=dropout_override,
    )

    if not os.path.exists(head_only_ckpt_path):
        raise FileNotFoundError(f"Head-only checkpoint not found: {head_only_ckpt_path}")

    state = torch.load(head_only_ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(device)
    return model, trait_cols, state


def predict_to_dataframe(
    model,
    dataloader,
    dataset: AESDataset,
    original_df: pd.DataFrame,
    trait_cols: List[str],
    score_ranges: Dict[str, Dict[str, Dict[str, float]]],
    global_trait_fallback: Dict[str, Dict[str, float]],
    device: torch.device,
    round_step: float,
    id_col: str,
    prompt_col: str,
    text_col: str,
) -> pd.DataFrame:
    model.eval()

    all_preds = []
    all_labels = []
    all_masks = []
    all_indices = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            label_mask = batch["label_mask"].to(device)
            idxs = batch["idx"].cpu().numpy()
            token_type_ids = batch["token_type_ids"].to(device) if "token_type_ids" in batch else None

            preds = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            all_masks.append(label_mask.detach().cpu().numpy())
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
    df = original_df.reset_index(drop=True)

    for row_pos, ds_idx in enumerate(indices):
        ds_idx = int(ds_idx)
        pid = normalize_prompt_id(dataset.prompt_ids[ds_idx])

        row = {
            "row_index": ds_idx,
            prompt_col: pid,
        }

        if id_col in df.columns:
            row[id_col] = df.at[ds_idx, id_col]
        if text_col in df.columns:
            row[text_col] = df.at[ds_idx, text_col]

        for j, trait in enumerate(trait_cols):
            if masks[row_pos, j] < 0.5:
                row[f"{trait}_gold"] = np.nan
                row[f"{trait}_pred_raw"] = np.nan
                row[f"{trait}_pred_default"] = np.nan
                continue

            rng = get_range_for_trait(score_ranges, pid, trait, global_trait_fallback)
            mn, mx = float(rng["min"]), float(rng["max"])

            gold_raw = denormalize_score(labels_norm[row_pos, j], mn, mx)
            pred_raw = denormalize_score(preds_norm[row_pos, j], mn, mx)

            gold_rounded = round_to_step(gold_raw, round_step)
            pred_default = round_to_step(pred_raw, round_step)

            row[f"{trait}_gold"] = float(np.clip(gold_rounded, mn, mx))
            row[f"{trait}_pred_raw"] = float(np.clip(pred_raw, mn, mx))
            row[f"{trait}_pred_default"] = float(np.clip(pred_default, mn, mx))

        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------
# Metrics and threshold-map application
# -----------------------------

def tune_thresholds_by_prompt_trait(
    dev_pred_df: pd.DataFrame,
    trait_cols: List[str],
    score_ranges: Dict[str, Dict[str, Dict[str, float]]],
    global_trait_fallback: Dict[str, Dict[str, float]],
    prompt_col: str,
    round_step: float,
    seed: int,
    grid_size: int,
    max_coord_iters: int,
    n_random_restarts: int,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    threshold_map: Dict[str, Dict[str, Dict[str, Any]]] = {}

    prompts = sorted(dev_pred_df[prompt_col].dropna().astype(str).map(normalize_prompt_id).unique().tolist())

    for prompt_id in prompts:
        prompt_df = dev_pred_df[dev_pred_df[prompt_col].astype(str).map(normalize_prompt_id) == prompt_id]
        threshold_map[prompt_id] = {}

        for trait in trait_cols:
            gold_col = f"{trait}_gold"
            pred_col = f"{trait}_pred_raw"
            if gold_col not in prompt_df.columns or pred_col not in prompt_df.columns:
                continue

            valid = prompt_df[[gold_col, pred_col]].dropna()
            if len(valid) == 0:
                continue

            rng = get_range_for_trait(score_ranges, prompt_id, trait, global_trait_fallback)
            mn, mx = float(rng["min"]), float(rng["max"])
            labels = valid_score_labels(mn, mx, round_step)

            result = tune_thresholds_for_one_trait(
                y_true_raw=valid[gold_col].to_numpy(dtype=np.float64),
                pred_raw=valid[pred_col].to_numpy(dtype=np.float64),
                labels=labels,
                step=round_step,
                seed=seed + hash((prompt_id, trait)) % 100000,
                grid_size=grid_size,
                max_coord_iters=max_coord_iters,
                n_random_restarts=n_random_restarts,
            )
            result["prompt_id"] = prompt_id
            result["trait"] = trait
            result["score_min"] = mn
            result["score_max"] = mx
            result["round_step"] = round_step
            threshold_map[prompt_id][trait] = result

    return threshold_map


def apply_threshold_map_to_dataframe(
    pred_df: pd.DataFrame,
    threshold_map: Dict[str, Dict[str, Dict[str, Any]]],
    trait_cols: List[str],
    score_ranges: Dict[str, Dict[str, Dict[str, float]]],
    global_trait_fallback: Dict[str, Dict[str, float]],
    prompt_col: str,
    round_step: float,
) -> pd.DataFrame:
    out = pred_df.copy()

    for idx, row in out.iterrows():
        prompt_id = normalize_prompt_id(row[prompt_col])

        for trait in trait_cols:
            pred_col = f"{trait}_pred_raw"
            tuned_col = f"{trait}_pred_tuned"

            if pred_col not in out.columns or pd.isna(row.get(pred_col, np.nan)):
                out.at[idx, tuned_col] = np.nan
                continue

            if prompt_id in threshold_map and trait in threshold_map[prompt_id]:
                item = threshold_map[prompt_id][trait]
                labels = np.asarray(item["labels"], dtype=np.float64)
                thresholds = np.asarray(item["thresholds"], dtype=np.float64)
            else:
                rng = get_range_for_trait(score_ranges, prompt_id, trait, global_trait_fallback)
                labels = valid_score_labels(float(rng["min"]), float(rng["max"]), round_step)
                thresholds = default_midpoint_thresholds(labels)

            tuned = apply_thresholds_to_raw_predictions(
                pred_raw=np.array([float(row[pred_col])], dtype=np.float64),
                thresholds=thresholds,
                labels=labels,
            )[0]
            out.at[idx, tuned_col] = float(tuned)

    return out


def compute_prediction_metrics(
    pred_df: pd.DataFrame,
    trait_cols: List[str],
    score_ranges: Dict[str, Dict[str, Dict[str, float]]],
    global_trait_fallback: Dict[str, Dict[str, float]],
    prompt_col: str,
    pred_suffix: str,
    round_step: float,
) -> Dict[str, Any]:
    trait_metrics: Dict[str, Any] = {}
    qwk_values = []
    rmse_values = []

    for trait in trait_cols:
        gold_col = f"{trait}_gold"
        pred_col = f"{trait}_{pred_suffix}"

        if gold_col not in pred_df.columns or pred_col not in pred_df.columns:
            trait_metrics[trait] = {"n": 0, "qwk": float("nan"), "rmse": float("nan")}
            continue

        valid = pred_df[[prompt_col, gold_col, pred_col]].dropna().copy()
        if len(valid) == 0:
            trait_metrics[trait] = {"n": 0, "qwk": float("nan"), "rmse": float("nan")}
            continue

        # If a single trait contains multiple prompts with different ranges,
        # compute QWK on normalized class indices per prompt, then concatenate indices.
        y_true_all = []
        y_pred_all = []
        y_true_raw_all = []
        y_pred_raw_all = []
        offset = 0

        for prompt_id, subdf in valid.groupby(valid[prompt_col].astype(str).map(normalize_prompt_id)):
            rng = get_range_for_trait(score_ranges, prompt_id, trait, global_trait_fallback)
            labels = valid_score_labels(float(rng["min"]), float(rng["max"]), round_step)

            y_true_raw = subdf[gold_col].to_numpy(dtype=np.float64)
            y_pred_raw = subdf[pred_col].to_numpy(dtype=np.float64)
            y_true_idx = scores_to_indices(y_true_raw, labels, round_step)
            y_pred_idx = scores_to_indices(y_pred_raw, labels, round_step)

            # Offset prompt-specific classes so sklearn treats them as separate ordered groups.
            # In the usual held-out setup there is only one prompt, so this does not change anything.
            y_true_all.extend((y_true_idx + offset).tolist())
            y_pred_all.extend((y_pred_idx + offset).tolist())
            offset += len(labels)

            y_true_raw_all.extend(y_true_raw.tolist())
            y_pred_raw_all.extend(y_pred_raw.tolist())

        y_true_all = np.asarray(y_true_all, dtype=int)
        y_pred_all = np.asarray(y_pred_all, dtype=int)
        y_true_raw_all = np.asarray(y_true_raw_all, dtype=np.float64)
        y_pred_raw_all = np.asarray(y_pred_raw_all, dtype=np.float64)

        if len(y_true_all) < 2 or len(np.unique(y_true_all)) < 2:
            qwk = float("nan")
        else:
            try:
                qwk = float(cohen_kappa_score(y_true_all, y_pred_all, weights="quadratic"))
            except Exception:
                qwk = float("nan")

        rmse = float(math.sqrt(mean_squared_error(y_true_raw_all, y_pred_raw_all)))

        if not math.isnan(qwk):
            qwk_values.append(qwk)
        if not math.isnan(rmse):
            rmse_values.append(rmse)

        trait_metrics[trait] = {
            "n": int(len(valid)),
            "qwk": qwk,
            "rmse": rmse,
        }

    return {
        "mean_qwk": float(np.mean(qwk_values)) if qwk_values else float("nan"),
        "mean_rmse": float(np.mean(rmse_values)) if rmse_values else float("nan"),
        "trait_metrics": trait_metrics,
    }


def print_short_metrics(name: str, metrics: Dict[str, Any]):
    print(f"{name}: mean_qwk={metrics['mean_qwk']:.6f} | mean_rmse={metrics['mean_rmse']:.6f}", flush=True)
    for trait, tm in metrics.get("trait_metrics", {}).items():
        if tm["n"] > 0:
            print(f"  - {trait}: n={tm['n']} qwk={tm['qwk']:.6f} rmse={tm['rmse']:.6f}", flush=True)




def build_trait_rows_for_summary(
    trait_cols: List[str],
    dev_default_metrics: Dict[str, Any],
    dev_tuned_metrics: Dict[str, Any],
    test_default_metrics: Dict[str, Any],
    test_tuned_metrics: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Create one row per trait for later averaging across repeats."""
    rows: List[Dict[str, Any]] = []

    for trait in trait_cols:
        dd = dev_default_metrics.get("trait_metrics", {}).get(trait, {})
        dt = dev_tuned_metrics.get("trait_metrics", {}).get(trait, {})
        td = test_default_metrics.get("trait_metrics", {}).get(trait, {})
        tt = test_tuned_metrics.get("trait_metrics", {}).get(trait, {})

        rows.append({
            "trait": trait,
            "dev_default_n": dd.get("n", 0),
            "dev_default_qwk": dd.get("qwk", float("nan")),
            "dev_default_rmse": dd.get("rmse", float("nan")),
            "dev_tuned_n": dt.get("n", 0),
            "dev_tuned_qwk": dt.get("qwk", float("nan")),
            "dev_tuned_rmse": dt.get("rmse", float("nan")),
            "test_default_n": td.get("n", 0),
            "test_default_qwk": td.get("qwk", float("nan")),
            "test_default_rmse": td.get("rmse", float("nan")),
            "test_tuned_n": tt.get("n", 0),
            "test_tuned_qwk": tt.get("qwk", float("nan")),
            "test_tuned_rmse": tt.get("rmse", float("nan")),
        })

    return rows


def save_traitwise_aggregate_csvs(
    trait_repeat_rows: List[Dict[str, Any]],
    output_root: str,
) -> None:
    """
    Save trait-wise repeat results and final mean-QWK reports.

    Main final report requested:
      final_trait_mean_qwk_report.csv

    This file has one row per heldout_prompt + fewshot_k + model_type, and one
    column per trait containing the mean TEST threshold-tuned QWK across repeats.
    Example columns:
      content_mean_qwk, organization_mean_qwk, ..., mean_qwk_across_available_traits
    """
    if not trait_repeat_rows:
        return

    trait_repeat_df = pd.DataFrame(trait_repeat_rows)

    # 1) Detailed file: every repeat, every trait.
    repeat_path = os.path.join(output_root, "threshold_tuning_traitwise_by_repeat.csv")
    trait_repeat_df.to_csv(repeat_path, index=False)

    group_cols = ["heldout_prompt", "fewshot_k", "model_type", "trait"]
    metric_cols = [
        "dev_default_qwk",
        "dev_tuned_qwk",
        "test_default_qwk",
        "test_tuned_qwk",
        "dev_default_rmse",
        "dev_tuned_rmse",
        "test_default_rmse",
        "test_tuned_rmse",
        "dev_default_n",
        "test_default_n",
        "dev_tuned_n",
        "test_tuned_n",
    ]
    metric_cols = [c for c in metric_cols if c in trait_repeat_df.columns]

    # 2) Long aggregate file: one row per trait, with mean/std across repeats.
    agg = trait_repeat_df.groupby(group_cols, dropna=False).agg(
        n_repeats=("repeat_name", "nunique"),
        **{f"{col}_mean": (col, "mean") for col in metric_cols},
        **{f"{col}_std": (col, "std") for col in metric_cols if not col.endswith("_n")},
    ).reset_index()

    preferred_order = [
        "heldout_prompt",
        "fewshot_k",
        "model_type",
        "trait",
        "n_repeats",
        "dev_default_qwk_mean",
        "dev_tuned_qwk_mean",
        "test_default_qwk_mean",
        "test_tuned_qwk_mean",
        "test_tuned_qwk_std",
        "dev_default_rmse_mean",
        "dev_tuned_rmse_mean",
        "test_default_rmse_mean",
        "test_tuned_rmse_mean",
        "test_tuned_rmse_std",
    ]
    ordered_cols = [c for c in preferred_order if c in agg.columns]
    remaining_cols = [c for c in agg.columns if c not in ordered_cols]
    agg = agg[ordered_cols + remaining_cols]

    agg_path = os.path.join(output_root, "threshold_tuning_traitwise_mean_across_repeats.csv")
    agg.to_csv(agg_path, index=False)

    # 3) Final compact report: one row per k/model, one column per trait.
    #    Values are the mean TEST threshold-tuned QWK across all repeats.
    final_base = agg[[
        "heldout_prompt",
        "fewshot_k",
        "model_type",
        "trait",
        "test_tuned_qwk_mean",
    ]].copy()

    wide = final_base.pivot_table(
        index=["heldout_prompt", "fewshot_k", "model_type"],
        columns="trait",
        values="test_tuned_qwk_mean",
        aggfunc="first",
    ).reset_index()

    # Rename trait columns to make the meaning explicit.
    non_trait_cols = {"heldout_prompt", "fewshot_k", "model_type"}
    rename_map = {c: f"{c}_mean_qwk" for c in wide.columns if c not in non_trait_cols}
    wide = wide.rename(columns=rename_map)

    # Add number of repeats used for each heldout/k/model.
    n_repeats = trait_repeat_df.groupby(
        ["heldout_prompt", "fewshot_k", "model_type"],
        dropna=False,
    )["repeat_name"].nunique().reset_index(name="n_repeats")

    wide = wide.merge(n_repeats, on=["heldout_prompt", "fewshot_k", "model_type"], how="left")

    # Put common trait columns in a stable order matching TRAIT_COLUMNS.
    trait_qwk_cols = [f"{t}_mean_qwk" for t in TRAIT_COLUMNS if f"{t}_mean_qwk" in wide.columns]
    other_qwk_cols = [c for c in wide.columns if c.endswith("_mean_qwk") and c not in trait_qwk_cols]
    wide["mean_qwk_across_available_traits"] = wide[trait_qwk_cols + other_qwk_cols].mean(axis=1, skipna=True)

    final_cols = ["heldout_prompt", "fewshot_k", "model_type", "n_repeats"] + trait_qwk_cols + other_qwk_cols + ["mean_qwk_across_available_traits"]
    wide = wide[[c for c in final_cols if c in wide.columns]]

    final_report_path = os.path.join(output_root, "final_trait_mean_qwk_report.csv")
    wide.to_csv(final_report_path, index=False)

    print(f"Saved trait-wise per-repeat CSV: {repeat_path}", flush=True)
    print(f"Saved trait-wise mean-across-repeats CSV: {agg_path}", flush=True)
    print(f"Saved final compact trait mean QWK report: {final_report_path}", flush=True)

# -----------------------------
# One model/run processing
# -----------------------------

def process_one_model(
    model_name_for_files: str,
    model,
    tokenizer,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: str,
    trait_cols: List[str],
    args,
    prompt_text_map: Dict[str, str],
    score_ranges: Dict[str, Dict[str, Dict[str, float]]],
    global_trait_fallback: Dict[str, Dict[str, float]],
    device: torch.device,
) -> Dict[str, Any]:
    ensure_dir(out_dir)

    dev_dataset = AESDataset(
        df=dev_df,
        tokenizer=tokenizer,
        trait_cols=trait_cols,
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
        trait_cols=trait_cols,
        prompt_col=args.prompt_col,
        text_col=args.text_col,
        prompt_text_map=prompt_text_map,
        score_ranges=score_ranges,
        global_trait_fallback=global_trait_fallback,
        max_length=args.max_length,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    dev_pred = predict_to_dataframe(
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
    test_pred = predict_to_dataframe(
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

    threshold_map = tune_thresholds_by_prompt_trait(
        dev_pred_df=dev_pred,
        trait_cols=trait_cols,
        score_ranges=score_ranges,
        global_trait_fallback=global_trait_fallback,
        prompt_col=args.prompt_col,
        round_step=args.round_step,
        seed=args.seed,
        grid_size=args.grid_size,
        max_coord_iters=args.max_coord_iters,
        n_random_restarts=args.n_random_restarts,
    )

    dev_tuned = apply_threshold_map_to_dataframe(
        pred_df=dev_pred,
        threshold_map=threshold_map,
        trait_cols=trait_cols,
        score_ranges=score_ranges,
        global_trait_fallback=global_trait_fallback,
        prompt_col=args.prompt_col,
        round_step=args.round_step,
    )
    test_tuned = apply_threshold_map_to_dataframe(
        pred_df=test_pred,
        threshold_map=threshold_map,
        trait_cols=trait_cols,
        score_ranges=score_ranges,
        global_trait_fallback=global_trait_fallback,
        prompt_col=args.prompt_col,
        round_step=args.round_step,
    )

    dev_default_metrics = compute_prediction_metrics(
        pred_df=dev_tuned,
        trait_cols=trait_cols,
        score_ranges=score_ranges,
        global_trait_fallback=global_trait_fallback,
        prompt_col=args.prompt_col,
        pred_suffix="pred_default",
        round_step=args.round_step,
    )
    test_default_metrics = compute_prediction_metrics(
        pred_df=test_tuned,
        trait_cols=trait_cols,
        score_ranges=score_ranges,
        global_trait_fallback=global_trait_fallback,
        prompt_col=args.prompt_col,
        pred_suffix="pred_default",
        round_step=args.round_step,
    )
    dev_tuned_metrics = compute_prediction_metrics(
        pred_df=dev_tuned,
        trait_cols=trait_cols,
        score_ranges=score_ranges,
        global_trait_fallback=global_trait_fallback,
        prompt_col=args.prompt_col,
        pred_suffix="pred_tuned",
        round_step=args.round_step,
    )
    test_tuned_metrics = compute_prediction_metrics(
        pred_df=test_tuned,
        trait_cols=trait_cols,
        score_ranges=score_ranges,
        global_trait_fallback=global_trait_fallback,
        prompt_col=args.prompt_col,
        pred_suffix="pred_tuned",
        round_step=args.round_step,
    )

    # Save outputs
    dev_tuned.to_csv(os.path.join(out_dir, f"{model_name_for_files}_dev_predictions_thresholded.csv"), index=False)
    test_tuned.to_csv(os.path.join(out_dir, f"{model_name_for_files}_test_predictions_thresholded.csv"), index=False)

    save_json(threshold_map, os.path.join(out_dir, f"{model_name_for_files}_thresholds.json"))
    save_json(dev_default_metrics, os.path.join(out_dir, f"{model_name_for_files}_dev_default_rounding_metrics.json"))
    save_json(test_default_metrics, os.path.join(out_dir, f"{model_name_for_files}_test_default_rounding_metrics.json"))
    save_json(dev_tuned_metrics, os.path.join(out_dir, f"{model_name_for_files}_dev_threshold_tuned_metrics.json"))
    save_json(test_tuned_metrics, os.path.join(out_dir, f"{model_name_for_files}_test_threshold_tuned_metrics.json"))

    print_short_metrics(f"{model_name_for_files} dev default", dev_default_metrics)
    print_short_metrics(f"{model_name_for_files} dev tuned", dev_tuned_metrics)
    print_short_metrics(f"{model_name_for_files} test default", test_default_metrics)
    print_short_metrics(f"{model_name_for_files} test tuned", test_tuned_metrics)

    trait_rows = build_trait_rows_for_summary(
        trait_cols=trait_cols,
        dev_default_metrics=dev_default_metrics,
        dev_tuned_metrics=dev_tuned_metrics,
        test_default_metrics=test_default_metrics,
        test_tuned_metrics=test_tuned_metrics,
    )

    return {
        "dev_default_mean_qwk": dev_default_metrics["mean_qwk"],
        "dev_tuned_mean_qwk": dev_tuned_metrics["mean_qwk"],
        "test_default_mean_qwk": test_default_metrics["mean_qwk"],
        "test_tuned_mean_qwk": test_tuned_metrics["mean_qwk"],
        "dev_default_mean_rmse": dev_default_metrics["mean_rmse"],
        "dev_tuned_mean_rmse": dev_tuned_metrics["mean_rmse"],
        "test_default_mean_rmse": test_default_metrics["mean_rmse"],
        "test_tuned_mean_rmse": test_tuned_metrics["mean_rmse"],
        "_trait_rows": trait_rows,
    }


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()
    ensure_dir(args.output_root)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    full_df = pd.read_csv(args.data_path, sep=args.sep)
    full_df[args.prompt_col] = full_df[args.prompt_col].apply(normalize_prompt_id)

    for trait in TRAIT_COLUMNS:
        if trait in full_df.columns:
            full_df[trait] = pd.to_numeric(full_df[trait], errors="coerce")

    all_prompts = sorted(full_df[args.prompt_col].astype(str).unique().tolist())
    heldout_prompts = parse_prompt_list(args.heldout_prompts, all_prompts)
    fewshot_sizes = sorted(parse_int_list(args.fewshot_sizes))

    prompt_text_map = build_prompt_text_map()
    score_ranges = build_score_ranges_from_hardcoded()
    global_trait_fallback = build_global_trait_fallback(full_df, TRAIT_COLUMNS)

    summary_rows: List[Dict[str, Any]] = []
    trait_repeat_rows: List[Dict[str, Any]] = []

    for heldout_prompt in heldout_prompts:
        heldout_prompt = normalize_prompt_id(heldout_prompt)
        split_prompt_dir = os.path.join(args.split_root, f"heldout_{heldout_prompt}")
        base_ckpt_dir = os.path.join(args.base_root, f"base_prompt{heldout_prompt}", "best_checkpoint")

        if not os.path.isdir(split_prompt_dir):
            print(f"Skipping heldout={heldout_prompt}: split dir not found -> {split_prompt_dir}", flush=True)
            continue
        if not os.path.isdir(base_ckpt_dir):
            print(f"Skipping heldout={heldout_prompt}: base checkpoint dir not found -> {base_ckpt_dir}", flush=True)
            continue

        repeat_dirs = sorted([
            os.path.join(split_prompt_dir, d)
            for d in os.listdir(split_prompt_dir)
            if d.startswith("repeat_") and os.path.isdir(os.path.join(split_prompt_dir, d))
        ])

        tokenizer = AutoTokenizer.from_pretrained(base_ckpt_dir, use_fast=True)

        for repeat_dir in repeat_dirs:
            repeat_name = os.path.basename(repeat_dir)
            print(f"\n=== heldout={heldout_prompt} | {repeat_name} ===", flush=True)

            dev_path = os.path.join(repeat_dir, "dev.tsv")
            test_path = os.path.join(repeat_dir, "test.tsv")
            if not os.path.exists(dev_path) or not os.path.exists(test_path):
                print(f"Skipping {repeat_dir}: dev.tsv or test.tsv missing", flush=True)
                continue

            dev_df = pd.read_csv(dev_path, sep="\t")
            test_df = pd.read_csv(test_path, sep="\t")
            dev_df[args.prompt_col] = dev_df[args.prompt_col].apply(normalize_prompt_id)
            test_df[args.prompt_col] = test_df[args.prompt_col].apply(normalize_prompt_id)

            repeat_out_dir = os.path.join(args.output_root, f"heldout_{heldout_prompt}", repeat_name)
            ensure_dir(repeat_out_dir)

            if args.include_zero_shot:
                print("\n--- zero-shot/base threshold tuning ---", flush=True)
                zero_model, zero_trait_cols, _ = load_base_checkpoint_into_model(
                    base_ckpt_dir=base_ckpt_dir,
                    device=device,
                    dropout_override=args.dropout_override,
                )

                zero_out_dir = os.path.join(repeat_out_dir, "zero_shot")
                zero_metrics = process_one_model(
                    model_name_for_files="zero_shot",
                    model=zero_model,
                    tokenizer=tokenizer,
                    dev_df=dev_df,
                    test_df=test_df,
                    out_dir=zero_out_dir,
                    trait_cols=zero_trait_cols,
                    args=args,
                    prompt_text_map=prompt_text_map,
                    score_ranges=score_ranges,
                    global_trait_fallback=global_trait_fallback,
                    device=device,
                )

                zero_trait_rows = zero_metrics.pop("_trait_rows", [])
                zero_context = {
                    "heldout_prompt": heldout_prompt,
                    "repeat_name": repeat_name,
                    "fewshot_k": "zero_shot",
                    "model_type": "base_zero_shot",
                }
                summary_rows.append({
                    **zero_context,
                    **zero_metrics,
                })
                for row in zero_trait_rows:
                    trait_repeat_rows.append({**zero_context, **row})

                del zero_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            for k in fewshot_sizes:
                run_dir = os.path.join(args.head_only_root, f"heldout_{heldout_prompt}", repeat_name, f"k_{k}")
                head_ckpt_path = os.path.join(run_dir, "best_head_only.pt")

                if not os.path.exists(head_ckpt_path):
                    print(f"Skipping k={k}: checkpoint not found -> {head_ckpt_path}", flush=True)
                    continue

                print(f"\n--- head-only threshold tuning | k={k} ---", flush=True)
                model, trait_cols, head_state = load_head_only_checkpoint_into_base_model(
                    base_ckpt_dir=base_ckpt_dir,
                    head_only_ckpt_path=head_ckpt_path,
                    device=device,
                    dropout_override=args.dropout_override,
                )

                out_dir = os.path.join(repeat_out_dir, f"k_{k}")
                metrics = process_one_model(
                    model_name_for_files="head_only",
                    model=model,
                    tokenizer=tokenizer,
                    dev_df=dev_df,
                    test_df=test_df,
                    out_dir=out_dir,
                    trait_cols=trait_cols,
                    args=args,
                    prompt_text_map=prompt_text_map,
                    score_ranges=score_ranges,
                    global_trait_fallback=global_trait_fallback,
                    device=device,
                )

                head_trait_rows = metrics.pop("_trait_rows", [])
                head_context = {
                    "heldout_prompt": heldout_prompt,
                    "repeat_name": repeat_name,
                    "fewshot_k": k,
                    "model_type": "head_only",
                    "best_epoch": head_state.get("best_epoch"),
                    "best_dev_mean_qwk_before_thresholding": head_state.get("best_dev_mean_qwk"),
                }
                summary_rows.append({
                    **head_context,
                    **metrics,
                })
                for row in head_trait_rows:
                    trait_repeat_rows.append({**head_context, **row})

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(args.output_root, "threshold_tuning_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved summary: {summary_path}", flush=True)
        save_traitwise_aggregate_csvs(trait_repeat_rows, args.output_root)
    else:
        print("\nNo threshold tuning runs were completed.", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
