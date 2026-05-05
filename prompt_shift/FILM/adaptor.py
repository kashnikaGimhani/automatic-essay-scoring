#!/usr/bin/env python3
"""
Zero-shot cross-prompt AES with trait-specific FiLM near trait heads and
an explicit essay-prompt matching objective.

What changed compared with the earlier script:
1. Trait-specific FiLM:
   - Each trait gets its own FiLM parameters from (prompt representation + trait embedding).
   - FiLM is applied immediately before each trait head.
2. Prompt-aware objective:
   - During training, each essay is trained to match its correct prompt metadata
     against a bank of source-prompt metadata representations.
   - Implemented as a contrastive essay->prompt classification loss.

Reads a TSV/CSV with columns like:
essay_id, essay_set, essay, overall, content, organization, ...

Example:
python zero_shot_film_traitmatch_asap.py \
  --data_path data/asap_traits.tsv \
  --heldout_prompt 1 \
  --output_dir runs/heldout_1_traitmatch \
  --model_name roberta-base \
  --prompt_meta_json asap_prompt_meta.json \
  --score_range_json asap_score_ranges.json \
  --exclude_columns target overall
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


DEFAULT_CANDIDATE_LABELS = [
    "overall",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--heldout_prompt", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--prompt_meta_json", type=str, default=None)
    parser.add_argument("--score_range_json", type=str, default=None)
    parser.add_argument("--text_column", type=str, default="essay")
    parser.add_argument("--prompt_column", type=str, default="essay_set")
    parser.add_argument("--id_column", type=str, default="essay_id")
    parser.add_argument("--candidate_label_columns", nargs="*", default=DEFAULT_CANDIDATE_LABELS)
    parser.add_argument("--exclude_columns", nargs="*", default=["target"])
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--prompt_max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--film_reg", type=float, default=1e-4)
    parser.add_argument("--match_loss_weight", type=float, default=0.1)
    parser.add_argument("--match_temperature", type=float, default=0.07)
    parser.add_argument("--contrastive_dim", type=int, default=256)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--share_prompt_encoder", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: Optional[str]) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_table(path: str) -> pd.DataFrame:
    if path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def detect_label_columns(df: pd.DataFrame, candidates: List[str], exclude: List[str]) -> List[str]:
    label_cols = []
    for col in candidates:
        if col in exclude or col not in df.columns:
            continue
        numeric_col = pd.to_numeric(df[col], errors="coerce")
        if numeric_col.notna().any():
            label_cols.append(col)
    if not label_cols:
        raise ValueError("No usable numeric score columns were found.")
    return label_cols


def safe_prompt_key(v) -> str:
    try:
        fv = float(v)
        if fv.is_integer():
            return str(int(fv))
    except Exception:
        pass
    return str(v)


def build_score_ranges(
    df: pd.DataFrame,
    prompt_col: str,
    label_cols: List[str],
    score_range_json: Optional[str],
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    external = load_json(score_range_json)
    if external:
        out = {}
        for prompt_id, trait_map in external.items():
            out[str(prompt_id)] = {
                str(trait): (float(bounds[0]), float(bounds[1]))
                for trait, bounds in trait_map.items()
            }
        return out

    out: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for prompt_id, group in df.groupby(prompt_col):
        prompt_key = safe_prompt_key(prompt_id)
        out[prompt_key] = {}
        for col in label_cols:
            numeric = pd.to_numeric(group[col], errors="coerce").dropna()
            if len(numeric) == 0:
                continue
            out[prompt_key][col] = (float(numeric.min()), float(numeric.max()))
    return out


def normalize_score(value: float, mn: float, mx: float) -> float:
    if math.isclose(mx, mn):
        return 0.0
    return (value - mn) / (mx - mn)


def denormalize_score(value: float, mn: float, mx: float) -> float:
    return value * (mx - mn) + mn


def build_prompt_texts(
    prompt_ids: List[str],
    label_cols: List[str],
    score_ranges: Dict[str, Dict[str, Tuple[float, float]]],
    prompt_meta_json: Optional[str],
) -> Dict[str, str]:
    extra_meta = load_json(prompt_meta_json)
    texts: Dict[str, str] = {}
    for prompt_id in prompt_ids:
        meta = extra_meta.get(str(prompt_id), {})
        description = str(meta.get("description", ""))
        rubric = str(meta.get("rubric", ""))
        genre = str(meta.get("genre", ""))
        trait_names = meta.get("traits", label_cols)
        ranges = score_ranges.get(str(prompt_id), {})

        range_lines = []
        for trait in trait_names:
            if trait in ranges:
                mn, mx = ranges[trait]
                range_lines.append(f"{trait}: {mn:g} to {mx:g}")
        range_text = "; ".join(range_lines) if range_lines else "unknown"

        parts = [
            f"Prompt ID: {prompt_id}",
            f"Description: {description}" if description else f"Description: Prompt {prompt_id}",
            f"Genre: {genre}" if genre else None,
            f"Traits: {', '.join(map(str, trait_names))}" if trait_names else None,
            f"Score ranges: {range_text}",
            f"Rubric: {rubric}" if rubric else None,
        ]
        texts[str(prompt_id)] = "\n".join([p for p in parts if p])
    return texts


@dataclass
class Example:
    essay_id: str
    prompt_id: str
    essay: str
    labels: np.ndarray
    mask: np.ndarray


class AESDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        prompt_texts: Dict[str, str],
        score_ranges: Dict[str, Dict[str, Tuple[float, float]]],
        label_cols: List[str],
        text_col: str,
        prompt_col: str,
        id_col: str,
        max_length: int,
        prompt_max_length: int,
    ):
        self.tokenizer = tokenizer
        self.label_cols = label_cols
        self.text_col = text_col
        self.prompt_col = prompt_col
        self.max_length = max_length
        self.prompt_max_length = prompt_max_length
        self.score_ranges = score_ranges
        self.examples: List[Example] = []
        self.prompt_token_cache = {}

        for prompt_id, prompt_text in prompt_texts.items():
            self.prompt_token_cache[str(prompt_id)] = self.tokenizer(
                prompt_text,
                truncation=True,
                padding="max_length",
                max_length=self.prompt_max_length,
                return_tensors="pt",
            )

        for _, row in df.iterrows():
            prompt_id = safe_prompt_key(row[prompt_col])
            labels, mask = self._build_labels_and_mask(row, prompt_id)
            self.examples.append(
                Example(
                    essay_id=str(row[id_col]),
                    prompt_id=prompt_id,
                    essay=str(row[text_col]),
                    labels=labels,
                    mask=mask,
                )
            )

    def _build_labels_and_mask(self, row: pd.Series, prompt_id: str) -> Tuple[np.ndarray, np.ndarray]:
        labels = np.zeros(len(self.label_cols), dtype=np.float32)
        mask = np.zeros(len(self.label_cols), dtype=np.float32)
        prompt_ranges = self.score_ranges.get(prompt_id, {})
        for i, col in enumerate(self.label_cols):
            val = pd.to_numeric(pd.Series([row.get(col, np.nan)]), errors="coerce").iloc[0]
            if pd.isna(val) or col not in prompt_ranges:
                continue
            mn, mx = prompt_ranges[col]
            labels[i] = normalize_score(float(val), mn, mx)
            mask[i] = 1.0
        return labels, mask

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        essay_tokens = self.tokenizer(
            ex.essay,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_tokens = self.prompt_token_cache[ex.prompt_id]
        return {
            "essay_id": ex.essay_id,
            "prompt_id": ex.prompt_id,
            "input_ids": essay_tokens["input_ids"].squeeze(0),
            "attention_mask": essay_tokens["attention_mask"].squeeze(0),
            "prompt_input_ids": prompt_tokens["input_ids"].squeeze(0),
            "prompt_attention_mask": prompt_tokens["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex.labels, dtype=torch.float),
            "label_mask": torch.tensor(ex.mask, dtype=torch.float),
        }


def collate_fn(batch: List[dict]) -> dict:
    out = {}
    for key in [
        "input_ids",
        "attention_mask",
        "prompt_input_ids",
        "prompt_attention_mask",
        "labels",
        "label_mask",
    ]:
        out[key] = torch.stack([x[key] for x in batch], dim=0)
    out["essay_id"] = [x["essay_id"] for x in batch]
    out["prompt_id"] = [x["prompt_id"] for x in batch]
    return out


class TraitSpecificFiLMAESModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        label_cols: List[str],
        dropout: float = 0.1,
        share_prompt_encoder: bool = False,
        contrastive_dim: int = 256,
        match_temperature: float = 0.07,
    ):
        super().__init__()
        self.label_cols = label_cols
        self.num_labels = len(label_cols)
        self.match_temperature = match_temperature

        self.essay_encoder = AutoModel.from_pretrained(model_name)
        self.prompt_encoder = self.essay_encoder if share_prompt_encoder else AutoModel.from_pretrained(model_name)
        hidden_size = self.essay_encoder.config.hidden_size

        self.prompt_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.essay_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.trait_embeddings = nn.Embedding(self.num_labels, hidden_size)
        self.condition_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gamma_head = nn.Linear(hidden_size, hidden_size)
        self.beta_head = nn.Linear(hidden_size, hidden_size)

        self.trait_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, 1),
                )
                for _ in range(self.num_labels)
            ]
        )

        self.essay_match_proj = nn.Linear(hidden_size, contrastive_dim)
        self.prompt_match_proj = nn.Linear(hidden_size, contrastive_dim)

    @staticmethod
    def pooled(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def encode_essay(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.essay_encoder(input_ids=input_ids, attention_mask=attention_mask)
        essay_repr = self.pooled(outputs.last_hidden_state, attention_mask)
        return self.essay_proj(essay_repr)

    def encode_prompt(self, prompt_input_ids: torch.Tensor, prompt_attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.prompt_encoder(input_ids=prompt_input_ids, attention_mask=prompt_attention_mask)
        prompt_repr = self.pooled(outputs.last_hidden_state, prompt_attention_mask)
        return self.prompt_proj(prompt_repr)

    def score_from_representations(self, essay_repr: torch.Tensor, prompt_repr: torch.Tensor) -> dict:
        batch_size, hidden_size = essay_repr.shape
        trait_emb = self.trait_embeddings.weight.unsqueeze(0).expand(batch_size, -1, -1)
        prompt_expand = prompt_repr.unsqueeze(1).expand(-1, self.num_labels, -1)
        cond = self.condition_proj(torch.cat([prompt_expand, trait_emb], dim=-1))

        gamma = 1.0 + 0.1 * torch.tanh(self.gamma_head(cond))
        beta = 0.1 * torch.tanh(self.beta_head(cond))
        modulated = gamma * essay_repr.unsqueeze(1) + beta

        pred_list = []
        for j, head in enumerate(self.trait_heads):
            pred_list.append(head(modulated[:, j, :]))
        preds = torch.cat(pred_list, dim=1)

        return {
            "preds": preds,
            "gamma": gamma,
            "beta": beta,
            "essay_repr": essay_repr,
            "prompt_repr": prompt_repr,
            "modulated": modulated,
        }

    def compute_matching_logits(self, essay_repr: torch.Tensor, prompt_bank_repr: torch.Tensor) -> torch.Tensor:
        essay_z = F.normalize(self.essay_match_proj(essay_repr), dim=-1)
        prompt_z = F.normalize(self.prompt_match_proj(prompt_bank_repr), dim=-1)
        return essay_z @ prompt_z.transpose(0, 1) / self.match_temperature

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        bank_input_ids: Optional[torch.Tensor] = None,
        bank_attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        essay_repr = self.encode_essay(input_ids, attention_mask)
        prompt_repr = self.encode_prompt(prompt_input_ids, prompt_attention_mask)
        outputs = self.score_from_representations(essay_repr, prompt_repr)
        if bank_input_ids is not None and bank_attention_mask is not None:
            bank_prompt_repr = self.encode_prompt(bank_input_ids, bank_attention_mask)
            outputs["match_logits"] = self.compute_matching_logits(essay_repr, bank_prompt_repr)
        return outputs


class MaskedMSELoss(nn.Module):
    def forward(self, preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sq = (preds - labels) ** 2
        sq = sq * mask
        denom = mask.sum().clamp(min=1.0)
        return sq.sum() / denom


def split_source_train_dev(
    source_df: pd.DataFrame,
    prompt_col: str,
    dev_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    train_parts = []
    dev_parts = []
    for _, group in source_df.groupby(prompt_col):
        idxs = list(group.index)
        rng.shuffle(idxs)
        dev_n = max(1, int(len(idxs) * dev_ratio)) if len(idxs) >= 10 else max(1, min(2, len(idxs) // 5))
        dev_idxs = set(idxs[:dev_n])
        train_parts.append(group.loc[[i for i in idxs if i not in dev_idxs]])
        dev_parts.append(group.loc[[i for i in idxs if i in dev_idxs]])
    return pd.concat(train_parts).reset_index(drop=True), pd.concat(dev_parts).reset_index(drop=True)


def safe_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return float("nan")
    try:
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except Exception:
        return float("nan")


def build_prompt_bank_tensors(
    tokenizer,
    prompt_texts: Dict[str, str],
    prompt_ids: List[str],
    prompt_max_length: int,
) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    input_ids = []
    attention_masks = []
    ordered_ids = []
    for pid in prompt_ids:
        tok = tokenizer(
            prompt_texts[pid],
            truncation=True,
            padding="max_length",
            max_length=prompt_max_length,
            return_tensors="pt",
        )
        ordered_ids.append(pid)
        input_ids.append(tok["input_ids"].squeeze(0))
        attention_masks.append(tok["attention_mask"].squeeze(0))
    bank = {
        "input_ids": torch.stack(input_ids, dim=0),
        "attention_mask": torch.stack(attention_masks, dim=0),
    }
    return ordered_ids, bank


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_cols: List[str],
    score_ranges: Dict[str, Dict[str, Tuple[float, float]]],
    film_reg: float,
) -> dict:
    model.eval()
    loss_fn = MaskedMSELoss()
    total_loss = 0.0
    steps = 0
    pred_store = {col: [] for col in label_cols}
    true_store = {col: [] for col in label_cols}
    rows = []

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                prompt_input_ids=batch["prompt_input_ids"].to(device),
                prompt_attention_mask=batch["prompt_attention_mask"].to(device),
            )
            preds = outputs["preds"]
            labels = batch["labels"].to(device)
            mask = batch["label_mask"].to(device)
            loss = loss_fn(preds, labels, mask)
            reg = film_reg * (((outputs["gamma"] - 1.0) ** 2).mean() + (outputs["beta"] ** 2).mean())
            total_loss += float((loss + reg).item())
            steps += 1

            preds_np = preds.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy()
            prompt_ids = batch["prompt_id"]
            essay_ids = batch["essay_id"]

            for row_idx in range(preds_np.shape[0]):
                row_record = {"essay_id": essay_ids[row_idx], "prompt_id": prompt_ids[row_idx]}
                prompt_key = str(prompt_ids[row_idx])
                prompt_ranges = score_ranges.get(prompt_key, {})
                for j, col in enumerate(label_cols):
                    if mask_np[row_idx, j] == 0 or col not in prompt_ranges:
                        row_record[f"pred_{col}"] = np.nan
                        row_record[f"true_{col}"] = np.nan
                        continue
                    mn, mx = prompt_ranges[col]
                    pred_raw = denormalize_score(float(preds_np[row_idx, j]), mn, mx)
                    true_raw = denormalize_score(float(labels_np[row_idx, j]), mn, mx)
                    pred_raw = float(np.clip(np.round(pred_raw), mn, mx))
                    true_raw = float(np.clip(np.round(true_raw), mn, mx))
                    row_record[f"pred_{col}"] = pred_raw
                    row_record[f"true_{col}"] = true_raw
                    pred_store[col].append(pred_raw)
                    true_store[col].append(true_raw)
                rows.append(row_record)

    metrics = {}
    qwk_values = []
    rmse_values = []
    for col in label_cols:
        y_true = np.array(true_store[col], dtype=np.float32)
        y_pred = np.array(pred_store[col], dtype=np.float32)
        if len(y_true) == 0:
            metrics[col] = {"n": 0, "qwk": float("nan"), "rmse": float("nan")}
            continue
        qwk = safe_qwk(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics[col] = {"n": int(len(y_true)), "qwk": qwk, "rmse": rmse}
        if not np.isnan(qwk):
            qwk_values.append(qwk)
        rmse_values.append(rmse)

    return {
        "loss": total_loss / max(steps, 1),
        "mean_qwk": float(np.mean(qwk_values)) if qwk_values else float("nan"),
        "mean_rmse": float(np.mean(rmse_values)) if rmse_values else float("nan"),
        "trait_metrics": metrics,
        "predictions": pd.DataFrame(rows),
    }


def train_one_run(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)
    set_seed(args.seed)

    df = read_table(args.data_path)
    for col in [args.text_column, args.prompt_column, args.id_column]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    label_cols = detect_label_columns(df, args.candidate_label_columns, args.exclude_columns)
    score_ranges = build_score_ranges(df, args.prompt_column, label_cols, args.score_range_json)

    df[args.prompt_column] = df[args.prompt_column].apply(safe_prompt_key)
    heldout_key = safe_prompt_key(args.heldout_prompt)
    source_df = df[df[args.prompt_column] != heldout_key].reset_index(drop=True)
    target_df = df[df[args.prompt_column] == heldout_key].reset_index(drop=True)
    if len(source_df) == 0 or len(target_df) == 0:
        raise ValueError("Source or target split is empty. Check heldout_prompt and essay_set values.")

    train_df, dev_df = split_source_train_dev(source_df, args.prompt_column, args.dev_ratio, args.seed)
    prompt_ids = sorted(df[args.prompt_column].astype(str).unique().tolist(), key=lambda x: int(x) if x.isdigit() else x)
    prompt_texts = build_prompt_texts(prompt_ids, label_cols, score_ranges, args.prompt_meta_json)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_ds = AESDataset(
        train_df, tokenizer, prompt_texts, score_ranges, label_cols,
        args.text_column, args.prompt_column, args.id_column, args.max_length, args.prompt_max_length
    )
    dev_ds = AESDataset(
        dev_df, tokenizer, prompt_texts, score_ranges, label_cols,
        args.text_column, args.prompt_column, args.id_column, args.max_length, args.prompt_max_length
    )
    test_ds = AESDataset(
        target_df, tokenizer, prompt_texts, score_ranges, label_cols,
        args.text_column, args.prompt_column, args.id_column, args.max_length, args.prompt_max_length
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    source_prompt_ids = sorted(source_df[args.prompt_column].astype(str).unique().tolist(), key=lambda x: int(x) if x.isdigit() else x)
    bank_prompt_ids, prompt_bank_cpu = build_prompt_bank_tensors(tokenizer, prompt_texts, source_prompt_ids, args.prompt_max_length)
    prompt_id_to_bank_index = {pid: idx for idx, pid in enumerate(bank_prompt_ids)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt_bank = {
        "input_ids": prompt_bank_cpu["input_ids"].to(device),
        "attention_mask": prompt_bank_cpu["attention_mask"].to(device),
    }

    model = TraitSpecificFiLMAESModel(
        model_name=args.model_name,
        label_cols=label_cols,
        dropout=args.dropout,
        share_prompt_encoder=args.share_prompt_encoder,
        contrastive_dim=args.contrastive_dim,
        match_temperature=args.match_temperature,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_training_steps = max(1, len(train_loader) * args.epochs)
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    score_loss_fn = MaskedMSELoss()
    match_loss_fn = nn.CrossEntropyLoss()

    best_dev_qwk = -1e9
    best_path = os.path.join(args.output_dir, "best_model.pt")
    history = []

    print("Starting training...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_total = 0.0
        running_score = 0.0
        running_match = 0.0
        running_reg = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                prompt_input_ids=batch["prompt_input_ids"].to(device),
                prompt_attention_mask=batch["prompt_attention_mask"].to(device),
                bank_input_ids=prompt_bank["input_ids"],
                bank_attention_mask=prompt_bank["attention_mask"],
            )
            labels = batch["labels"].to(device)
            mask = batch["label_mask"].to(device)

            score_loss = score_loss_fn(outputs["preds"], labels, mask)
            reg = args.film_reg * (((outputs["gamma"] - 1.0) ** 2).mean() + (outputs["beta"] ** 2).mean())
            match_targets = torch.tensor([prompt_id_to_bank_index[pid] for pid in batch["prompt_id"]], dtype=torch.long, device=device)
            match_loss = match_loss_fn(outputs["match_logits"], match_targets)
            total = score_loss + reg + args.match_loss_weight * match_loss
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_total += float(total.item())
            running_score += float(score_loss.item())
            running_match += float(match_loss.item())
            running_reg += float(reg.item())

        dev_metrics = evaluate(model, dev_loader, device, label_cols, score_ranges, args.film_reg)
        record = {
            "epoch": epoch,
            "train_total_loss": running_total / max(len(train_loader), 1),
            "train_score_loss": running_score / max(len(train_loader), 1),
            "train_match_loss": running_match / max(len(train_loader), 1),
            "train_reg_loss": running_reg / max(len(train_loader), 1),
            "dev_loss": dev_metrics["loss"],
            "dev_mean_qwk": dev_metrics["mean_qwk"],
            "dev_mean_rmse": dev_metrics["mean_rmse"],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        current_qwk = dev_metrics["mean_qwk"]
        if np.isnan(current_qwk):
            current_qwk = -1e9
        if current_qwk > best_dev_qwk:
            best_dev_qwk = current_qwk
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_cols": label_cols,
                    "score_ranges": score_ranges,
                    "prompt_texts": prompt_texts,
                    "args": vars(args),
                    "epoch": epoch,
                    "best_dev_mean_qwk": dev_metrics["mean_qwk"],
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    final_dev = evaluate(model, dev_loader, device, label_cols, score_ranges, args.film_reg)
    final_test = evaluate(model, test_loader, device, label_cols, score_ranges, args.film_reg)

    final_dev["predictions"].to_csv(os.path.join(args.output_dir, "dev_predictions.csv"), index=False)
    final_test["predictions"].to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)
    pd.DataFrame(history).to_csv(os.path.join(args.output_dir, "training_history.csv"), index=False)

    metrics_summary = {
        "heldout_prompt": heldout_key,
        "label_columns": label_cols,
        "best_epoch": int(ckpt["epoch"]),
        "best_dev_mean_qwk": ckpt["best_dev_mean_qwk"],
        "final_dev_mean_qwk": final_dev["mean_qwk"],
        "final_dev_mean_rmse": final_dev["mean_rmse"],
        "final_test_mean_qwk": final_test["mean_qwk"],
        "final_test_mean_rmse": final_test["mean_rmse"],
        "dev_trait_metrics": final_dev["trait_metrics"],
        "test_trait_metrics": final_test["trait_metrics"],
        "match_loss_weight": args.match_loss_weight,
        "match_temperature": args.match_temperature,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print("Saved outputs to:", args.output_dir)
    print(json.dumps(metrics_summary, indent=2))


if __name__ == "__main__":
    train_one_run(parse_args())
