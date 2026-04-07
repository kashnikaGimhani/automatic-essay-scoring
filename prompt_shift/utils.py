import os
import json
import math
import random
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from transformers import AutoModel


MODEL_NAME = "roberta-base"

TRAIT_COLUMNS = [
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

PROMPT_DESCRIPTIONS = {
    1: "Write a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you.",
    2: "Write a persuasive essay to a newspaper reflecting your views on censorship in libraries. Support your position with convincing arguments from your own experience, observations, and/or reading.",
    3: "Write a response that explains how the features of the setting affect the cyclist. In your response, include examples from the source material that support your conclusion.",
    4: "Write a response that explains why the author concludes the story with the last paragraph of the source material given. In your response, include details and examples from the story that support your ideas.",
    5: "Describe the mood created by the author in the memoir given in the source material. Support your answer with relevant and specific information from the memoir.",
    6: "Based on the excerpt given in the source material, describe the obstacles the builders of the Empire State Building faced in attempting to allow dirigibles to dock there. Support your answer with relevant and specific information from the excerpt.",
    7: "Do only one of the following: write a story about a time when you were patient OR write a story about a time when someone you know was patient OR write a story in your own way about patience.",
    8: "We all understand the benefits of laughter. For example, someone once said, 'Laughter is the shortest distance between two people.' Many other people believe that laughter is an important part of any relationship. Tell a true story in which laughter was one element or part.",
}

# Kept exactly aligned with the uploaded base training script.
SCORE_RANGES = {
    ("1", "content"): (1.0, 6.0),
    ("1", "organization"): (1.0, 6.0),
    ("1", "word_choice"): (1.0, 6.0),
    ("1", "sentence_fluency"): (1.0, 6.0),
    ("1", "conventions"): (1.0, 6.0),

    ("2", "content"): (1.0, 6.0),
    ("2", "organization"): (1.0, 6.0),
    ("2", "word_choice"): (1.0, 6.0),
    ("2", "sentence_fluency"): (1.0, 6.0),
    ("2", "conventions"): (1.0, 6.0),

    ("3", "content"): (0.0, 3.0),
    ("3", "prompt_adherence"): (0.0, 3.0),
    ("3", "language"): (0.0, 3.0),
    ("3", "narrativity"): (0.0, 3.0),

    ("4", "content"): (0.0, 3.0),
    ("4", "prompt_adherence"): (0.0, 3.0),
    ("4", "language"): (0.0, 3.0),
    ("4", "narrativity"): (0.0, 3.0),

    ("5", "content"): (0.0, 4.0),
    ("5", "prompt_adherence"): (0.0, 4.0),
    ("5", "language"): (0.0, 4.0),
    ("5", "narrativity"): (0.0, 4.0),

    ("6", "content"): (0.0, 4.0),
    ("6", "prompt_adherence"): (0.0, 4.0),
    ("6", "language"): (0.0, 4.0),
    ("6", "narrativity"): (0.0, 4.0),

    ("7", "content"): (0.0, 6.0),
    ("7", "organization"): (0.0, 6.0),
    ("7", "style"): (0.0, 6.0),
    ("7", "conventions"): (0.0, 6.0),

    ("8", "content"): (2.0, 12.0),
    ("8", "organization"): (2.0, 12.0),
    ("8", "voice"): (2.0, 12.0),
    ("8", "word_choice"): (2.0, 12.0),
    ("8", "sentence_fluency"): (2.0, 12.0),
    ("8", "conventions"): (2.0, 12.0),
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_prompt_id(x: Any) -> str:
    s = str(x).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def build_prompt_text_map() -> Dict[str, str]:
    return {normalize_prompt_id(pid): text for pid, text in PROMPT_DESCRIPTIONS.items()}


def build_score_ranges_from_hardcoded() -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for (prompt_id, trait), (mn, mx) in SCORE_RANGES.items():
        if prompt_id not in out:
            out[prompt_id] = {}
        out[prompt_id][trait] = {"min": float(mn), "max": float(mx)}
    return out


def build_global_trait_fallback(
    df: pd.DataFrame,
    trait_cols: List[str],
) -> Dict[str, Dict[str, float]]:
    fallback = {}
    for trait in trait_cols:
        if trait not in df.columns:
            continue
        vals = pd.to_numeric(df[trait], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        fallback[trait] = {"min": float(vals.min()), "max": float(vals.max())}
    return fallback


def get_range_for_trait(
    score_ranges: Dict[str, Dict[str, Dict[str, float]]],
    prompt_id: str,
    trait: str,
    global_fallback: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    if prompt_id in score_ranges and trait in score_ranges[prompt_id]:
        return score_ranges[prompt_id][trait]
    if global_fallback is not None and trait in global_fallback:
        return global_fallback[trait]
    raise KeyError(f"No score range found for prompt={prompt_id}, trait={trait}")


def normalize_score(x: float, mn: float, mx: float) -> float:
    if mx <= mn:
        return 0.0
    return (x - mn) / (mx - mn)


def denormalize_score(x: float, mn: float, mx: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    if mx <= mn:
        return mn
    return x * (mx - mn) + mn


def round_to_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return round(x / step) * step


def split_source_by_prompt(
    df_source: pd.DataFrame,
    prompt_col: str,
    dev_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    dev_parts = []

    rng = np.random.RandomState(seed)

    for _, subdf in df_source.groupby(prompt_col):
        idx = np.arange(len(subdf))
        rng.shuffle(idx)

        n_dev = max(1, int(round(len(subdf) * dev_ratio))) if len(subdf) > 1 else 0
        dev_idx = idx[:n_dev]
        train_idx = idx[n_dev:]

        subdf = subdf.reset_index(drop=True)

        if len(train_idx) == 0 and len(dev_idx) > 0:
            train_idx = dev_idx[:1]
            dev_idx = dev_idx[1:]

        train_parts.append(subdf.iloc[train_idx].copy())
        if len(dev_idx) > 0:
            dev_parts.append(subdf.iloc[dev_idx].copy())

    train_df = pd.concat(train_parts, axis=0).reset_index(drop=True)
    dev_df = (
        pd.concat(dev_parts, axis=0).reset_index(drop=True)
        if dev_parts else pd.DataFrame(columns=df_source.columns)
    )
    return train_df, dev_df


class AESDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        trait_cols: List[str],
        prompt_col: str,
        text_col: str,
        prompt_text_map: Dict[str, str],
        score_ranges: Dict[str, Dict[str, Dict[str, float]]],
        global_trait_fallback: Dict[str, Dict[str, float]],
        max_length: int,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.trait_cols = trait_cols
        self.prompt_col = prompt_col
        self.text_col = text_col
        self.score_ranges = score_ranges
        self.global_trait_fallback = global_trait_fallback

        self.prompt_ids = [normalize_prompt_id(x) for x in self.df[prompt_col].tolist()]
        prompt_texts = [
            prompt_text_map.get(pid, f"Prompt {pid}")
            for pid in self.prompt_ids
        ]
        essay_texts = self.df[text_col].fillna("").astype(str).tolist()

        enc = tokenizer(
            prompt_texts,
            essay_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.token_type_ids = enc["token_type_ids"] if "token_type_ids" in enc else None

        n = len(self.df)
        t = len(trait_cols)

        self.labels_raw = np.full((n, t), np.nan, dtype=np.float32)
        self.labels_norm = np.zeros((n, t), dtype=np.float32)
        self.label_mask = np.zeros((n, t), dtype=np.float32)

        for i in range(n):
            pid = self.prompt_ids[i]
            for j, trait in enumerate(trait_cols):
                if trait not in self.df.columns:
                    continue
                val = pd.to_numeric(pd.Series([self.df.at[i, trait]]), errors="coerce").iloc[0]
                if pd.isna(val):
                    continue

                val = float(val)
                rng = get_range_for_trait(score_ranges, pid, trait, global_trait_fallback)
                mn, mx = rng["min"], rng["max"]

                self.labels_raw[i, j] = val
                self.labels_norm[i, j] = normalize_score(val, mn, mx)
                self.label_mask[i, j] = 1.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": torch.tensor(self.labels_norm[idx], dtype=torch.float32),
            "label_mask": torch.tensor(self.label_mask[idx], dtype=torch.float32),
            "idx": torch.tensor(idx, dtype=torch.long),
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        return item


def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


class MultiTraitAESModel(nn.Module):
    def __init__(self, dropout: float, num_traits: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.encoder.config.hidden_size

        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_traits),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
        )
        pooled = masked_mean_pool(outputs.last_hidden_state, attention_mask)
        preds = self.regressor(pooled)
        return preds


def masked_regression_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    if loss_type == "mse":
        loss = (preds - targets) ** 2
    elif loss_type == "huber":
        diff = torch.abs(preds - targets)
        quadratic = torch.minimum(diff, torch.tensor(huber_delta, device=diff.device))
        linear = diff - quadratic
        loss = 0.5 * quadratic ** 2 + huber_delta * linear
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    loss = loss * mask
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom


def quadratic_weighted_kappa_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) < 2:
        return float("nan")
    if len(np.unique(y_true)) < 2:
        return float("nan")

    try:
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except Exception:
        return float("nan")


def evaluate(
    model: nn.Module,
    dataloader,
    dataset: AESDataset,
    trait_cols: List[str],
    score_ranges: Dict[str, Dict[str, Dict[str, float]]],
    global_trait_fallback: Dict[str, Dict[str, float]],
    device: torch.device,
    round_step: float,
    loss_type: str,
    huber_delta: float,
) -> Dict[str, Any]:
    model.eval()

    all_preds = []
    all_labels = []
    all_masks = []
    all_indices = []

    total_loss = 0.0
    total_steps = 0

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

            loss = masked_regression_loss(
                preds=preds,
                targets=labels,
                mask=label_mask,
                loss_type=loss_type,
                huber_delta=huber_delta,
            )
            total_loss += loss.item()
            total_steps += 1

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_masks.append(label_mask.cpu().numpy())
            all_indices.append(idxs)

    if not all_preds:
        return {
            "loss": float("nan"),
            "mean_qwk": float("nan"),
            "mean_rmse": float("nan"),
            "trait_metrics": {},
        }

    preds_norm = np.concatenate(all_preds, axis=0)
    labels_norm = np.concatenate(all_labels, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    indices = np.concatenate(all_indices, axis=0)

    order = np.argsort(indices)
    preds_norm = preds_norm[order]
    labels_norm = labels_norm[order]
    masks = masks[order]
    indices = indices[order]

    trait_gold: Dict[str, List[float]] = {t: [] for t in trait_cols}
    trait_pred: Dict[str, List[float]] = {t: [] for t in trait_cols}

    for row_pos, ds_idx in enumerate(indices):
        pid = dataset.prompt_ids[int(ds_idx)]

        for j, trait in enumerate(trait_cols):
            if masks[row_pos, j] < 0.5:
                continue

            rng = get_range_for_trait(score_ranges, pid, trait, global_trait_fallback)
            mn, mx = rng["min"], rng["max"]

            gold_raw = denormalize_score(labels_norm[row_pos, j], mn, mx)
            pred_raw = denormalize_score(preds_norm[row_pos, j], mn, mx)

            gold_rounded = round_to_step(gold_raw, round_step)
            pred_rounded = round_to_step(pred_raw, round_step)

            gold_rounded = float(np.clip(gold_rounded, mn, mx))
            pred_rounded = float(np.clip(pred_rounded, mn, mx))

            trait_gold[trait].append(gold_rounded)
            trait_pred[trait].append(pred_rounded)

    trait_metrics = {}
    qwk_values = []
    rmse_values = []

    for trait in trait_cols:
        y_true = np.array(trait_gold[trait], dtype=np.float32)
        y_pred = np.array(trait_pred[trait], dtype=np.float32)

        if len(y_true) == 0:
            trait_metrics[trait] = {
                "n": 0,
                "qwk": float("nan"),
                "rmse": float("nan"),
            }
            continue

        qwk = quadratic_weighted_kappa_safe(y_true, y_pred)
        rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))

        if not math.isnan(qwk):
            qwk_values.append(qwk)
        if not math.isnan(rmse):
            rmse_values.append(rmse)

        trait_metrics[trait] = {
            "n": int(len(y_true)),
            "qwk": qwk,
            "rmse": rmse,
        }

    mean_qwk = float(np.mean(qwk_values)) if qwk_values else float("nan")
    mean_rmse = float(np.mean(rmse_values)) if rmse_values else float("nan")

    return {
        "loss": total_loss / max(total_steps, 1),
        "mean_qwk": mean_qwk,
        "mean_rmse": mean_rmse,
        "trait_metrics": trait_metrics,
    }


def format_metrics_for_print(name: str, metrics: Dict[str, Any]) -> str:
    lines = [
        f"{name}:",
        f"  loss      = {metrics.get('loss', float('nan')):.6f}",
        f"  mean_qwk  = {metrics.get('mean_qwk', float('nan')):.6f}",
        f"  mean_rmse = {metrics.get('mean_rmse', float('nan')):.6f}",
    ]
    for trait, tm in metrics.get("trait_metrics", {}).items():
        lines.append(
            f"  - {trait}: n={tm['n']} qwk={tm['qwk']:.6f} rmse={tm['rmse']:.6f}"
            if tm["n"] > 0 else
            f"  - {trait}: n=0 qwk=nan rmse=nan"
        )
    return "\n".join(lines)
