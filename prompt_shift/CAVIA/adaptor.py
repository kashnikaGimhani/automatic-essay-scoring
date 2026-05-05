#!/usr/bin/env python3
"""
CAVIA-style prompt context adaptation for multi-trait AES with rubric-aware
context initialization.

What this script does
---------------------
1. Meta-trains on source prompts.
2. Learns a shared essay encoder + shared trait heads.
3. Learns a small prompt-context vector phi that is adapted per prompt/task.
4. Initializes phi either from zeros (plain CAVIA) or from a rubric/prompt-spec
   embedding (rubric-aware extension).
5. Adapts only phi on a held-out prompt using k-shot support essays.
6. Reports trait-level and mean QWK / RMSE, plus optional overall-from-traits.

Main design choices
-------------------
- Task = prompt.
- Inner loop updates ONLY phi.
- Outer loop updates encoder, FiLM generators, trait heads, and optional
  rubric-aware phi initializer.
- FiLM conditioning is applied near the regression heads, which is usually a
  good starting point for AES.

Expected input files
--------------------
1) data TSV/CSV with at least:
   - an essay text column
   - a prompt id column
   - trait score columns

2) score_ranges JSON, e.g.
{
  "1": {
    "content": [1, 6],
    "organization": [1, 6],
    "word_choice": [1, 6],
    "sentence_fluency": [1, 6],
    "conventions": [1, 6]
  },
  "2": {
    "content": {"min": 1, "max": 6},
    "organization": {"min": 1, "max": 6}
  }
}

3) prompt_specs JSON, e.g.
{
  "1": {
    "title": "Narrative writing",
    "description": "Write a personal narrative ...",
    "rubric": "High scores reflect ...",
    "traits": {
      "content": "Ideas are relevant and developed.",
      "organization": "Logical sequencing and cohesion."
    },
    "score_ranges": {
      "content": "1-6",
      "organization": "1-6"
    }
  }
}

Example usage
-------------
python cavia_rubric_aes.py \
  --data_path data/asap_all_traits.tsv \
  --sep '\t' \
  --prompt_col essay_set \
  --text_col essay \
  --traits content organization word_choice sentence_fluency conventions \
  --heldout_prompt 2 \
  --score_ranges_json data/score_ranges.json \
  --prompt_specs_json data/prompt_specs.json \
  --output_dir results/cavia_rubric_heldout2 \
  --encoder_name roberta-base \
  --max_length 512 \
  --meta_steps 1000 \
  --meta_batch_tasks 4 \
  --support_size 16 \
  --query_size 16 \
  --inner_steps 3 \
  --inner_lr 0.1 \
  --context_dim 32 \
  --repeats 5 \
  --k_values 8 16 32 64 128
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_table(path: str, sep: str) -> pd.DataFrame:
    if path.endswith('.csv'):
        return pd.read_csv(path)
    return pd.read_csv(path, sep=sep)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prompt_to_str(x: Any) -> str:
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def parse_range_item(item: Any) -> Tuple[float, float]:
    if isinstance(item, dict):
        if 'min' in item and 'max' in item:
            return float(item['min']), float(item['max'])
        raise ValueError(f'Invalid range dict: {item}')
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return float(item[0]), float(item[1])
    raise ValueError(f'Invalid score range item: {item}')


def normalize_score(value: float, min_v: float, max_v: float) -> float:
    if pd.isna(value):
        return np.nan
    denom = max(max_v - min_v, 1e-8)
    return float((value - min_v) / denom)


def denormalize_score(value: float, min_v: float, max_v: float) -> float:
    return float(min_v + value * (max_v - min_v))


def clip_round_score(value: float, min_v: float, max_v: float) -> int:
    if value is None or not np.isfinite(value):
        mid = (float(min_v) + float(max_v)) / 2.0
        value = mid
    return int(np.clip(np.rint(value), min_v, max_v))


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sq = (pred - target) ** 2
    sq = sq * mask.float()
    denom = mask.float().sum().clamp(min=1.0)
    return sq.sum() / denom


def grad_clip_tensor(g: torch.Tensor, max_norm: Optional[float]) -> torch.Tensor:
    if max_norm is None:
        return g
    norm = torch.norm(g)
    if torch.isfinite(norm) and norm > max_norm:
        g = g * (max_norm / (norm + 1e-8))
    return g


def infer_all_traits(
    score_ranges: Dict[str, Dict[str, Tuple[float, float]]],
    prompt_specs: Dict[str, Dict[str, Any]],
    df: pd.DataFrame,
    traits_arg: Optional[List[str]],
) -> List[str]:
    if traits_arg:
        return list(dict.fromkeys([t for t in traits_arg if t != 'overall']))

    ordered: List[str] = []
    seen = set()

    for pid in sorted(score_ranges.keys(), key=lambda x: (str(x))):
        for t in score_ranges[pid].keys():
            if t != 'overall' and t not in seen:
                ordered.append(t)
                seen.add(t)

    for pid in sorted(prompt_specs.keys(), key=lambda x: (str(x))):
        for t in prompt_specs.get(pid, {}).get('traits', []) or []:
            if t != 'overall' and t not in seen:
                ordered.append(t)
                seen.add(t)

    if ordered:
        return ordered

    candidate_cols = [
        c for c in df.columns
        if c not in {'essay', 'essay_set', 'prompt_id', 'id'} and pd.api.types.is_numeric_dtype(df[c])
    ]
    for c in candidate_cols:
        if c != 'overall' and c not in seen:
            ordered.append(c)
            seen.add(c)

    return ordered


def get_prompt_active_traits(
    prompt_id: str,
    all_traits: List[str],
    score_ranges: Dict[str, Dict[str, Tuple[float, float]]],
    prompt_specs: Dict[str, Dict[str, Any]],
) -> List[str]:
    ordered: List[str] = []
    seen = set()

    if prompt_id in prompt_specs:
        for t in prompt_specs[prompt_id].get('traits', []) or []:
            if t != 'overall' and t in all_traits and t not in seen:
                ordered.append(t)
                seen.add(t)

    if prompt_id in score_ranges:
        for t in score_ranges[prompt_id].keys():
            if t != 'overall' and t in all_traits and t not in seen:
                ordered.append(t)
                seen.add(t)

    if not ordered:
        ordered = [t for t in all_traits if prompt_id in score_ranges and t in score_ranges[prompt_id]]

    return ordered


def build_prompt_spec_text(
    prompt_id: str,
    spec: Dict[str, Any],
    active_traits: List[str],
    score_ranges: Dict[str, Dict[str, Tuple[float, float]]],
) -> str:
    parts: List[str] = [f"Prompt ID: {prompt_id}"]

    for field_name, label in [
        ('description', 'Description'),
        ('rubric', 'Rubric'),
        ('genre', 'Genre'),
        ('prompt_type', 'Prompt type'),
    ]:
        if spec.get(field_name):
            parts.append(f"{label}: {spec[field_name]}")

    if 'source_dependent' in spec:
        parts.append(f"Source dependent: {bool(spec['source_dependent'])}")

    trait_guidance = spec.get('trait_guidance', {}) or {}
    trait_lines = []
    for t in active_traits:
        t_parts = [f"Trait: {t}"]
        if prompt_id in score_ranges and t in score_ranges[prompt_id]:
            lo, hi = score_ranges[prompt_id][t]
            t_parts.append(f"Range: {lo} to {hi}")
        guidance = trait_guidance.get(t, {}) or {}
        for key, label in [
            ('definition', 'Definition'),
            ('high_score', 'High score'),
            ('mid_score', 'Mid score'),
            ('low_score', 'Low score'),
        ]:
            if guidance.get(key):
                t_parts.append(f"{label}: {guidance[key]}")
        trait_lines.append(" | ".join(t_parts))
    if trait_lines:
        parts.append("Trait details: " + " || ".join(trait_lines))

    scoring_policy = spec.get('scoring_policy', {}) or {}
    if scoring_policy:
        policy_text = "; ".join([f"{k}={v}" for k, v in scoring_policy.items()])
        parts.append(f"Scoring policy: {policy_text}")

    special_notes = spec.get('special_notes', {}) or {}
    if special_notes:
        notes_text = "; ".join([f"{k}={v}" for k, v in special_notes.items()])
        parts.append(f"Special notes: {notes_text}")

    return "\n".join(parts)


# -----------------------------
# Data prep
# -----------------------------

@dataclass
class Batch:
    texts: List[str]
    prompt_ids: List[str]
    targets_raw: np.ndarray        # [B, T]
    targets_norm: np.ndarray       # [B, T]
    mask: np.ndarray               # [B, T]


class AESDataModule:
    def __init__(
        self,
        df: pd.DataFrame,
        prompt_col: str,
        text_col: str,
        traits: List[str],
        score_ranges: Dict[str, Dict[str, Tuple[float, float]]],
    ):
        self.df = df.copy()
        self.prompt_col = prompt_col
        self.text_col = text_col
        self.traits = traits
        self.score_ranges = score_ranges

        self.df[self.prompt_col] = self.df[self.prompt_col].map(prompt_to_str)
        self.df[self.text_col] = self.df[self.text_col].astype(str)

        self._check_columns()
        self._add_normalized_targets()

    def _check_columns(self) -> None:
        missing = [c for c in [self.prompt_col, self.text_col] if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        for t in self.traits:
            if t not in self.df.columns:
                self.df[t] = np.nan

    def _add_normalized_targets(self) -> None:
        for t in self.traits:
            norm_vals = []
            for _, row in self.df.iterrows():
                pid = row[self.prompt_col]
                raw_value = row[t] if t in row else np.nan
                if pd.isna(raw_value) or pid not in self.score_ranges or t not in self.score_ranges[pid]:
                    norm_vals.append(np.nan)
                    continue
                lo, hi = self.score_ranges[pid][t]
                norm_vals.append(normalize_score(raw_value, lo, hi))
            self.df[f"{t}__norm"] = norm_vals

    def prompts(self) -> List[str]:
        return sorted(self.df[self.prompt_col].unique().tolist())

    def by_prompt(self, prompt_id: str) -> pd.DataFrame:
        return self.df[self.df[self.prompt_col] == prompt_id].reset_index(drop=True)

    def make_batch(self, sub_df: pd.DataFrame) -> Batch:
        targets_raw = sub_df[self.traits].to_numpy(dtype=np.float32)
        targets_norm = sub_df[[f"{t}__norm" for t in self.traits]].to_numpy(dtype=np.float32)
        mask = (~np.isnan(targets_raw)).astype(np.float32)
        return Batch(
            texts=sub_df[self.text_col].astype(str).tolist(),
            prompt_ids=sub_df[self.prompt_col].tolist(),
            targets_raw=targets_raw,
            targets_norm=targets_norm,
            mask=mask,
        )


def split_source_prompt_pool(
    df: pd.DataFrame,
    prompt_col: str,
    prompt_id: str,
    dev_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sub = df[df[prompt_col] == prompt_id].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(sub)
    n_dev = max(1, int(round(n * dev_frac)))
    dev_df = sub.iloc[:n_dev].reset_index(drop=True)
    train_df = sub.iloc[n_dev:].reset_index(drop=True)
    return train_df, dev_df


def split_heldout_prompt(
    df: pd.DataFrame,
    prompt_col: str,
    heldout_prompt: str,
    dev_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sub = df[df[prompt_col] == heldout_prompt].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(sub)
    n_test = max(1, int(round(n * test_frac)))
    n_dev = max(1, int(round(n * dev_frac)))
    if n_test + n_dev >= n:
        raise ValueError("Held-out prompt split leaves no training examples. Reduce dev/test fraction.")
    test_df = sub.iloc[:n_test].reset_index(drop=True)
    dev_df = sub.iloc[n_test:n_test + n_dev].reset_index(drop=True)
    train_df = sub.iloc[n_test + n_dev:].reset_index(drop=True)
    return train_df, dev_df, test_df


def sample_support_query(
    pool_df: pd.DataFrame,
    support_size: int,
    query_size: int,
    rng: random.Random,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(pool_df)
    if n < support_size + query_size:
        # sample with replacement if needed
        support_idx = [rng.randrange(n) for _ in range(support_size)]
        query_idx = [rng.randrange(n) for _ in range(query_size)]
        return pool_df.iloc[support_idx].reset_index(drop=True), pool_df.iloc[query_idx].reset_index(drop=True)
    idxs = list(range(n))
    rng.shuffle(idxs)
    support_idx = idxs[:support_size]
    query_idx = idxs[support_size:support_size + query_size]
    return pool_df.iloc[support_idx].reset_index(drop=True), pool_df.iloc[query_idx].reset_index(drop=True)


# -----------------------------
# Model
# -----------------------------

class MeanPooler(nn.Module):
    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-8)
        return summed / denom


class PromptContextAESModel(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        traits: List[str],
        context_dim: int = 32,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        context_init_mode: str = "rubric",
        freeze_encoder: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.traits = traits
        self.context_dim = context_dim
        self.context_init_mode = context_init_mode

        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder_hidden = self.encoder.config.hidden_size
        self.pooler = MeanPooler()

        if hasattr(self.encoder.config, 'use_cache'):
            self.encoder.config.use_cache = False
        if gradient_checkpointing and hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable()

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.essay_proj = nn.Sequential(
            nn.Linear(self.encoder_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Rubric/spec embedding -> initial context phi0
        self.spec_to_context = nn.Sequential(
            nn.Linear(self.encoder_hidden, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, context_dim),
        )

        # Shared FiLM conditioned on phi.
        self.shared_gamma = nn.Linear(context_dim, hidden_dim)
        self.shared_beta = nn.Linear(context_dim, hidden_dim)

        # Trait-specific FiLM + heads.
        self.trait_gamma = nn.ModuleDict({t: nn.Linear(context_dim, hidden_dim) for t in traits})
        self.trait_beta = nn.ModuleDict({t: nn.Linear(context_dim, hidden_dim) for t in traits})
        self.trait_heads = nn.ModuleDict(
            {
                t: nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for t in traits
            }
        )

    def encode_essay(self, essay_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.encoder(**essay_inputs)
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = self.pooler(out.last_hidden_state, essay_inputs['attention_mask'])
        return self.essay_proj(pooled)

    def initial_context(self, spec_embedding: torch.Tensor) -> torch.Tensor:
        # spec_embedding: [D] or [1, D]
        if spec_embedding.ndim == 1:
            spec_embedding = spec_embedding.unsqueeze(0)
        if self.context_init_mode == 'zero':
            phi0 = torch.zeros((1, self.context_dim), device=spec_embedding.device, dtype=spec_embedding.dtype)
        elif self.context_init_mode == 'rubric':
            phi0 = self.spec_to_context(spec_embedding)
        else:
            raise ValueError(f"Unknown context_init_mode: {self.context_init_mode}")
        phi0.requires_grad_(True)
        return phi0

    def forward_with_context(self, essay_inputs: Dict[str, torch.Tensor], phi: torch.Tensor) -> torch.Tensor:
        h = self.encode_essay(essay_inputs)  # [B, H]

        shared_gamma = self.shared_gamma(phi)  # [1, H]
        shared_beta = self.shared_beta(phi)    # [1, H]
        h = (1.0 + shared_gamma) * h + shared_beta

        outputs = []
        for t in self.traits:
            tg = self.trait_gamma[t](phi)
            tb = self.trait_beta[t](phi)
            ht = (1.0 + tg) * h + tb
            logit = self.trait_heads[t](ht).squeeze(-1)
            pred = torch.sigmoid(logit)  # normalized score in [0, 1]
            outputs.append(pred)
        return torch.stack(outputs, dim=1)  # [B, T]


# -----------------------------
# Tokenization helpers
# -----------------------------

def tokenize_texts(tokenizer: AutoTokenizer, texts: List[str], max_length: int, device: torch.device) -> Dict[str, torch.Tensor]:
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    return {k: v.to(device) for k, v in enc.items()}


def batch_to_tensors(batch: Batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    targets = torch.tensor(batch.targets_norm, dtype=torch.float32, device=device)
    mask = torch.tensor(batch.mask, dtype=torch.float32, device=device)
    return targets, mask


def get_autocast_context(device: torch.device, amp_dtype: str):
    if device.type != 'cuda' or amp_dtype == 'none':
        return nullcontext()
    if amp_dtype == 'bf16':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    raise ValueError(f"Unsupported amp_dtype: {amp_dtype}")


# -----------------------------
# Metrics
# -----------------------------

def denorm_predictions_for_prompt(
    preds_norm: np.ndarray,
    prompt_id: str,
    traits: List[str],
    score_ranges: Dict[str, Dict[str, Tuple[float, float]]],
) -> np.ndarray:
    preds_raw = np.zeros_like(preds_norm, dtype=np.float32)
    for j, t in enumerate(traits):
        lo, hi = score_ranges[prompt_id][t]
        preds_raw[:, j] = lo + preds_norm[:, j] * (hi - lo)
    return preds_raw


def compute_metrics_single_prompt(
    y_true_raw: np.ndarray,
    y_pred_norm: np.ndarray,
    prompt_id: str,
    traits: List[str],
    score_ranges: Dict[str, Dict[str, Tuple[float, float]]],
    include_overall: bool = True,
) -> Dict[str, Any]:
    active_traits = [t for t in traits if prompt_id in score_ranges and t in score_ranges[prompt_id]]
    active_indices = [traits.index(t) for t in active_traits]

    trait_metrics: Dict[str, Dict[str, float]] = {}
    qwk_vals: List[float] = []
    rmse_vals: List[float] = []

    for j_full, t in zip(active_indices, active_traits):
        base_mask = ~np.isnan(y_true_raw[:, j_full])
        lo, hi = score_ranges[prompt_id][t]
        yp_all = lo + y_pred_norm[:, j_full] * (hi - lo)
        finite_mask = np.isfinite(yp_all)
        mask = base_mask & finite_mask
        yt = y_true_raw[mask, j_full]
        yp = yp_all[mask]
        yp_round = np.array([clip_round_score(v, lo, hi) for v in yp], dtype=np.int32) if len(yp) else np.array([], dtype=np.int32)
        yt_round = np.array([clip_round_score(v, lo, hi) for v in yt], dtype=np.int32) if len(yt) else np.array([], dtype=np.int32)

        if len(yt) == 0:
            qwk = 0.0
            rmse = float('nan')
        else:
            if len(np.unique(yt_round)) < 2 or len(np.unique(yp_round)) < 1:
                qwk = 0.0
            else:
                try:
                    qwk = float(cohen_kappa_score(yt_round, yp_round, weights='quadratic'))
                except Exception:
                    qwk = 0.0
            rmse = float(math.sqrt(mean_squared_error(yt, yp)))

        trait_metrics[t] = {'n': int(mask.sum()), 'qwk': qwk, 'rmse': rmse}
        if not math.isnan(rmse):
            rmse_vals.append(rmse)
        qwk_vals.append(qwk)

    out: Dict[str, Any] = {
        'active_traits': active_traits,
        'trait_metrics': trait_metrics,
        'mean_qwk': float(np.mean(qwk_vals)) if qwk_vals else 0.0,
        'mean_rmse': float(np.mean(rmse_vals)) if rmse_vals else float('nan'),
    }

    if include_overall and active_indices:
        true_active = y_true_raw[:, active_indices]
        pred_raw_active = np.zeros_like(true_active, dtype=np.float32)
        for local_j, t in enumerate(active_traits):
            lo, hi = score_ranges[prompt_id][t]
            pred_raw_active[:, local_j] = lo + y_pred_norm[:, active_indices[local_j]] * (hi - lo)

        true_overall = np.nanmean(true_active, axis=1)
        pred_overall = np.nanmean(pred_raw_active, axis=1)
        valid = np.isfinite(true_overall) & np.isfinite(pred_overall)
        true_overall = true_overall[valid]
        pred_overall = pred_overall[valid]
        if len(true_overall) > 0:
            lo = float(np.nanmin(true_overall))
            hi = float(np.nanmax(true_overall))
            true_overall_round = np.rint(true_overall).astype(int)
            pred_overall_round = np.clip(np.rint(pred_overall), np.floor(lo), np.ceil(hi)).astype(int)
            if len(np.unique(true_overall_round)) < 2 or len(np.unique(pred_overall_round)) < 1:
                overall_qwk = 0.0
            else:
                try:
                    overall_qwk = float(cohen_kappa_score(true_overall_round, pred_overall_round, weights='quadratic'))
                except Exception:
                    overall_qwk = 0.0
            overall_rmse = float(math.sqrt(mean_squared_error(true_overall, pred_overall)))
            out['overall_from_traits'] = {
                'qwk': overall_qwk,
                'rmse': overall_rmse,
            }

    return out


# -----------------------------
# Prompt spec embedding cache
# -----------------------------

@torch.no_grad()
def precompute_prompt_spec_embeddings(
    prompt_specs: Dict[str, Dict[str, Any]],
    prompt_ids: List[str],
    traits: List[str],
    score_ranges: Dict[str, Dict[str, Tuple[float, float]]],
    spec_encoder_name: str,
    max_length: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(spec_encoder_name, use_fast=True)
    encoder = AutoModel.from_pretrained(spec_encoder_name).to(device)
    encoder.eval()

    embeddings: Dict[str, torch.Tensor] = {}
    for pid in prompt_ids:
        if pid not in prompt_specs:
            raise ValueError(f"Prompt spec missing for prompt_id={pid}")
        active_traits = get_prompt_active_traits(pid, traits, score_ranges, prompt_specs)
        spec_text = build_prompt_spec_text(pid, prompt_specs[pid], active_traits, score_ranges)
        enc = tokenizer(
            [spec_text],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = encoder(**enc)
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            pooled = out.pooler_output[0]
        else:
            mask = enc['attention_mask'].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
            pooled = pooled[0]
        embeddings[pid] = pooled.detach().cpu()

    del encoder
    torch.cuda.empty_cache()
    return embeddings


# -----------------------------
# Inner-loop / outer-loop routines
# -----------------------------

def adaptation_step(
    model: PromptContextAESModel,
    tokenizer: AutoTokenizer,
    batch: Batch,
    phi: torch.Tensor,
    max_length: int,
    device: torch.device,
    grad_clip: Optional[float],
    create_graph: bool,
    inner_lr: float,
    amp_dtype: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    essay_inputs = tokenize_texts(tokenizer, batch.texts, max_length, device)
    targets, mask = batch_to_tensors(batch, device)
    with get_autocast_context(device, amp_dtype):
        preds = model.forward_with_context(essay_inputs, phi)
        preds = torch.nan_to_num(preds.float(), nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        loss = masked_mse_loss(preds, targets, mask)
    loss_for_grad = loss
    if not torch.isfinite(loss_for_grad):
        safe_loss = torch.zeros((), device=device, dtype=phi.dtype)
        return torch.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0), safe_loss
    grad_phi = torch.autograd.grad(loss_for_grad, phi, create_graph=create_graph, retain_graph=create_graph, allow_unused=False)[0]
    grad_phi = torch.nan_to_num(grad_phi, nan=0.0, posinf=0.0, neginf=0.0)
    grad_phi = grad_clip_tensor(grad_phi, grad_clip)
    new_phi = phi - inner_lr * grad_phi
    new_phi = torch.nan_to_num(new_phi, nan=0.0, posinf=0.0, neginf=0.0)
    return new_phi, loss


def predict_batch(
    model: PromptContextAESModel,
    tokenizer: AutoTokenizer,
    batch: Batch,
    phi: torch.Tensor,
    max_length: int,
    device: torch.device,
    amp_dtype: str,
) -> Tuple[np.ndarray, float]:
    essay_inputs = tokenize_texts(tokenizer, batch.texts, max_length, device)
    targets, mask = batch_to_tensors(batch, device)
    with get_autocast_context(device, amp_dtype):
        preds = model.forward_with_context(essay_inputs, phi)
        preds = torch.nan_to_num(preds.float(), nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        loss_t = masked_mse_loss(preds, targets, mask)
        if not torch.isfinite(loss_t):
            loss_t = torch.tensor(float('nan'), device=device)
    return preds.detach().cpu().numpy(), float(loss_t.detach().cpu().item())


def run_validation_episodes(
    model: PromptContextAESModel,
    tokenizer: AutoTokenizer,
    data_module: AESDataModule,
    source_dev_pools: Dict[str, pd.DataFrame],
    prompt_spec_embeddings: Dict[str, torch.Tensor],
    traits: List[str],
    score_ranges: Dict[str, Dict[str, Tuple[float, float]]],
    support_size: int,
    query_size: int,
    inner_steps: int,
    inner_lr: float,
    val_episodes_per_prompt: int,
    max_length: int,
    device: torch.device,
    grad_clip: Optional[float],
    seed: int,
    amp_dtype: str,
) -> Dict[str, Any]:
    model.eval()
    rng = random.Random(seed)
    metrics_all = []

    for pid, pool in source_dev_pools.items():
        if len(pool) < 2:
            continue
        for _ in range(val_episodes_per_prompt):
            support_df, query_df = sample_support_query(pool, support_size, query_size, rng)
            support_batch = data_module.make_batch(support_df)
            query_batch = data_module.make_batch(query_df)

            spec_emb = prompt_spec_embeddings[pid].to(device)
            phi = model.initial_context(spec_emb)
            for _step in range(inner_steps):
                phi, _ = adaptation_step(
                    model=model,
                    tokenizer=tokenizer,
                    batch=support_batch,
                    phi=phi,
                    max_length=max_length,
                    device=device,
                    grad_clip=grad_clip,
                    create_graph=False,
                    inner_lr=inner_lr,
                    amp_dtype=amp_dtype,
                )

            with torch.no_grad():
                preds_norm, _ = predict_batch(model, tokenizer, query_batch, phi, max_length, device, amp_dtype)
            m = compute_metrics_single_prompt(
                y_true_raw=query_batch.targets_raw,
                y_pred_norm=preds_norm,
                prompt_id=pid,
                traits=traits,
                score_ranges=score_ranges,
                include_overall=True,
            )
            metrics_all.append(m)

    if not metrics_all:
        return {'mean_qwk': 0.0, 'mean_rmse': float('nan')}

    return aggregate_metric_dicts(metrics_all, traits)


def aggregate_metric_dicts(metrics_all: List[Dict[str, Any]], traits: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {'trait_metrics': {t: {'n': 0, 'qwk_vals': [], 'rmse_vals': []} for t in traits}}
    mean_qwks = []
    mean_rmses = []
    overall_qwks = []
    overall_rmses = []

    for m in metrics_all:
        mean_qwks.append(m.get('mean_qwk', 0.0))
        if not math.isnan(m.get('mean_rmse', float('nan'))):
            mean_rmses.append(m['mean_rmse'])
        for t in traits:
            tm = m['trait_metrics'].get(t)
            if tm is None:
                continue
            out['trait_metrics'][t]['n'] += tm['n']
            out['trait_metrics'][t]['qwk_vals'].append(tm['qwk'])
            if not math.isnan(tm['rmse']):
                out['trait_metrics'][t]['rmse_vals'].append(tm['rmse'])
        if 'overall_from_traits' in m:
            overall_qwks.append(m['overall_from_traits']['qwk'])
            overall_rmses.append(m['overall_from_traits']['rmse'])

    final_trait_metrics = {}
    for t in traits:
        tm = out['trait_metrics'][t]
        final_trait_metrics[t] = {
            'n': int(tm['n']),
            'qwk': float(np.mean(tm['qwk_vals'])) if tm['qwk_vals'] else 0.0,
            'rmse': float(np.mean(tm['rmse_vals'])) if tm['rmse_vals'] else float('nan'),
        }

    final = {
        'trait_metrics': final_trait_metrics,
        'mean_qwk': float(np.mean(mean_qwks)) if mean_qwks else 0.0,
        'mean_rmse': float(np.mean(mean_rmses)) if mean_rmses else float('nan'),
    }
    if overall_qwks:
        final['overall_from_traits'] = {
            'qwk': float(np.mean(overall_qwks)),
            'rmse': float(np.mean(overall_rmses)),
        }
    return final


# -----------------------------
# Meta-training
# -----------------------------

def meta_train(
    model: PromptContextAESModel,
    tokenizer: AutoTokenizer,
    data_module: AESDataModule,
    source_train_pools: Dict[str, pd.DataFrame],
    source_dev_pools: Dict[str, pd.DataFrame],
    prompt_spec_embeddings: Dict[str, torch.Tensor],
    traits: List[str],
    score_ranges: Dict[str, Dict[str, Tuple[float, float]]],
    output_dir: str,
    meta_steps: int,
    meta_batch_tasks: int,
    support_size: int,
    query_size: int,
    inner_steps: int,
    inner_lr: float,
    meta_lr: float,
    weight_decay: float,
    warmup_ratio: float,
    max_length: int,
    device: torch.device,
    val_every: int,
    val_episodes_per_prompt: int,
    grad_clip: Optional[float],
    first_order: bool,
    seed: int,
    amp_dtype: str,
) -> Dict[str, Any]:
    model.to(device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=meta_lr, weight_decay=weight_decay)
    warmup_steps = int(meta_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=meta_steps)

    source_prompt_ids = sorted(source_train_pools.keys())
    rng = random.Random(seed)

    best_state = None
    best_val_qwk = -1e9
    history = []

    pbar = tqdm(range(1, meta_steps + 1), desc='Meta-training')
    for step in pbar:
        model.train()
        optimizer.zero_grad()
        task_losses = []

        for _ in range(meta_batch_tasks):
            pid = rng.choice(source_prompt_ids)
            pool = source_train_pools[pid]
            support_df, query_df = sample_support_query(pool, support_size, query_size, rng)
            support_batch = data_module.make_batch(support_df)
            query_batch = data_module.make_batch(query_df)

            spec_emb = prompt_spec_embeddings[pid].to(device)
            phi = model.initial_context(spec_emb)

            for _inner in range(inner_steps):
                phi, _ = adaptation_step(
                    model=model,
                    tokenizer=tokenizer,
                    batch=support_batch,
                    phi=phi,
                    max_length=max_length,
                    device=device,
                    grad_clip=grad_clip,
                    create_graph=not first_order,
                    inner_lr=inner_lr,
                    amp_dtype=amp_dtype,
                )

            essay_inputs_q = tokenize_texts(tokenizer, query_batch.texts, max_length, device)
            targets_q, mask_q = batch_to_tensors(query_batch, device)
            with get_autocast_context(device, amp_dtype):
                preds_q = model.forward_with_context(essay_inputs_q, phi)
                query_loss = masked_mse_loss(preds_q.float(), targets_q, mask_q)
            task_losses.append(query_loss)
            del support_batch, query_batch, essay_inputs_q, targets_q, mask_q, preds_q, query_loss, spec_emb, phi

        meta_loss = torch.stack(task_losses).mean()
        meta_loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()
        scheduler.step()

        log_row = {
            'step': step,
            'meta_loss': float(meta_loss.detach().cpu().item()),
            'lr': float(scheduler.get_last_lr()[0]),
        }

        if step % val_every == 0 or step == meta_steps:
            val_metrics = run_validation_episodes(
                model=model,
                tokenizer=tokenizer,
                data_module=data_module,
                source_dev_pools=source_dev_pools,
                prompt_spec_embeddings=prompt_spec_embeddings,
                traits=traits,
                score_ranges=score_ranges,
                support_size=support_size,
                query_size=query_size,
                inner_steps=inner_steps,
                inner_lr=inner_lr,
                val_episodes_per_prompt=val_episodes_per_prompt,
                max_length=max_length,
                device=device,
                grad_clip=grad_clip,
                seed=seed + step,
                amp_dtype=amp_dtype,
            )
            log_row['val_mean_qwk'] = val_metrics['mean_qwk']
            log_row['val_mean_rmse'] = val_metrics['mean_rmse']
            pbar.set_postfix(loss=f"{log_row['meta_loss']:.4f}", val_qwk=f"{val_metrics['mean_qwk']:.4f}")

            if val_metrics['mean_qwk'] > best_val_qwk:
                best_val_qwk = val_metrics['mean_qwk']
                best_state = {
                    'model_state_dict': copy.deepcopy(model.state_dict()),
                    'step': step,
                    'val_metrics': val_metrics,
                }
                torch.save(best_state, os.path.join(output_dir, 'best_meta_checkpoint.pt'))
        else:
            pbar.set_postfix(loss=f"{log_row['meta_loss']:.4f}")

        history.append(log_row)

    if best_state is None:
        best_state = {
            'model_state_dict': copy.deepcopy(model.state_dict()),
            'step': meta_steps,
            'val_metrics': {'mean_qwk': 0.0, 'mean_rmse': float('nan')},
        }
        torch.save(best_state, os.path.join(output_dir, 'best_meta_checkpoint.pt'))

    save_json(history, os.path.join(output_dir, 'meta_history.json'))
    save_json(best_state['val_metrics'], os.path.join(output_dir, 'best_meta_val_metrics.json'))

    return {
        'best_step': best_state['step'],
        'best_val_metrics': best_state['val_metrics'],
        'history': history,
    }


# -----------------------------
# Held-out prompt adaptation / evaluation
# -----------------------------

def adapt_and_select_phi(
    model: PromptContextAESModel,
    tokenizer: AutoTokenizer,
    data_module: AESDataModule,
    prompt_id: str,
    support_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    prompt_spec_embedding: torch.Tensor,
    traits: List[str],
    score_ranges: Dict[str, Dict[str, Tuple[float, float]]],
    max_length: int,
    device: torch.device,
    adapt_steps: int,
    inner_lr: float,
    grad_clip: Optional[float],
    amp_dtype: str,
) -> Dict[str, Any]:
    model.eval()
    support_batch = data_module.make_batch(support_df)
    dev_batch = data_module.make_batch(dev_df)

    phi = model.initial_context(prompt_spec_embedding.to(device))
    phi_candidates: List[torch.Tensor] = [phi.detach().clone()]
    dev_metrics_all: List[Dict[str, Any]] = []

    with torch.no_grad():
        preds0, _ = predict_batch(model, tokenizer, dev_batch, phi, max_length, device, amp_dtype)
    dev_metrics0 = compute_metrics_single_prompt(
        y_true_raw=dev_batch.targets_raw,
        y_pred_norm=preds0,
        prompt_id=prompt_id,
        traits=traits,
        score_ranges=score_ranges,
        include_overall=True,
    )
    dev_metrics_all.append(dev_metrics0)

    for _ in range(adapt_steps):
        phi, _ = adaptation_step(
            model=model,
            tokenizer=tokenizer,
            batch=support_batch,
            phi=phi,
            max_length=max_length,
            device=device,
            grad_clip=grad_clip,
            create_graph=False,
            inner_lr=inner_lr,
            amp_dtype=amp_dtype,
        )
        phi_candidates.append(phi.detach().clone())
        with torch.no_grad():
            preds_dev, _ = predict_batch(model, tokenizer, dev_batch, phi, max_length, device, amp_dtype)
        dev_metrics = compute_metrics_single_prompt(
            y_true_raw=dev_batch.targets_raw,
            y_pred_norm=preds_dev,
            prompt_id=prompt_id,
            traits=traits,
            score_ranges=score_ranges,
            include_overall=True,
        )
        dev_metrics_all.append(dev_metrics)

    best_step = int(np.argmax([m['mean_qwk'] for m in dev_metrics_all]))
    best_phi = phi_candidates[best_step]

    return {
        'best_step': best_step,
        'best_phi': best_phi,
        'dev_metrics_by_step': dev_metrics_all,
        'best_dev_metrics': dev_metrics_all[best_step],
        'zero_shot_dev_metrics': dev_metrics_all[0],
    }


def evaluate_with_phi(
    model: PromptContextAESModel,
    tokenizer: AutoTokenizer,
    data_module: AESDataModule,
    prompt_id: str,
    eval_df: pd.DataFrame,
    phi: torch.Tensor,
    traits: List[str],
    score_ranges: Dict[str, Dict[str, Tuple[float, float]]],
    max_length: int,
    device: torch.device,
    amp_dtype: str,
) -> Dict[str, Any]:
    eval_batch = data_module.make_batch(eval_df)
    model.eval()
    with torch.no_grad():
        preds_norm, loss = predict_batch(model, tokenizer, eval_batch, phi.to(device), max_length, device, amp_dtype)
    metrics = compute_metrics_single_prompt(
        y_true_raw=eval_batch.targets_raw,
        y_pred_norm=preds_norm,
        prompt_id=prompt_id,
        traits=traits,
        score_ranges=score_ranges,
        include_overall=True,
    )
    metrics['loss'] = loss
    return metrics


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--sep', type=str, default='\t')
    p.add_argument('--prompt_col', type=str, default='essay_set')
    p.add_argument('--text_col', type=str, default='essay')
    p.add_argument('--traits', nargs='*', default=None)
    p.add_argument('--heldout_prompt', type=str, required=True)
    p.add_argument('--score_ranges_json', type=str, required=True)
    p.add_argument('--prompt_specs_json', type=str, required=True)
    p.add_argument('--output_dir', type=str, required=True)

    p.add_argument('--encoder_name', type=str, default='roberta-base')
    p.add_argument('--spec_encoder_name', type=str, default=None)
    p.add_argument('--max_length', type=int, default=384)
    p.add_argument('--freeze_encoder', action='store_true')
    p.add_argument('--gradient_checkpointing', action='store_true')
    p.add_argument('--amp_dtype', type=str, default='bf16', choices=['none', 'bf16'])

    p.add_argument('--context_dim', type=int, default=32)
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--context_init_mode', type=str, default='rubric', choices=['zero', 'rubric'])

    p.add_argument('--meta_steps', type=int, default=1000)
    p.add_argument('--meta_batch_tasks', type=int, default=2)
    p.add_argument('--support_size', type=int, default=8)
    p.add_argument('--query_size', type=int, default=8)
    p.add_argument('--inner_steps', type=int, default=3)
    p.add_argument('--inner_lr', type=float, default=0.1)
    p.add_argument('--meta_lr', type=float, default=2e-5)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_ratio', type=float, default=0.06)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--first_order', action='store_true')

    p.add_argument('--source_dev_frac', type=float, default=0.15)
    p.add_argument('--heldout_dev_frac', type=float, default=0.2)
    p.add_argument('--heldout_test_frac', type=float, default=0.2)
    p.add_argument('--val_every', type=int, default=50)
    p.add_argument('--val_episodes_per_prompt', type=int, default=4)

    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--k_values', nargs='+', type=int, default=[8, 16, 32, 64, 128])
    p.add_argument('--adapt_steps', type=int, default=10)

    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Save config for reproducibility.
    save_json(vars(args), os.path.join(args.output_dir, 'config.json'))

    df = read_table(args.data_path, args.sep)
    score_ranges_raw = load_json(args.score_ranges_json)
    prompt_specs = load_json(args.prompt_specs_json)

    score_ranges: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for pid_raw, trait_map in score_ranges_raw.items():
        pid = prompt_to_str(pid_raw)
        score_ranges[pid] = {}
        for t, item in trait_map.items():
            score_ranges[pid][t] = parse_range_item(item)

    all_traits = infer_all_traits(score_ranges, prompt_specs, df, args.traits)
    print(f"Using trait union: {all_traits}")

    data_module = AESDataModule(
        df=df,
        prompt_col=args.prompt_col,
        text_col=args.text_col,
        traits=all_traits,
        score_ranges=score_ranges,
    )

    all_prompts = data_module.prompts()
    heldout_prompt = prompt_to_str(args.heldout_prompt)
    if heldout_prompt not in all_prompts:
        raise ValueError(f"Held-out prompt {heldout_prompt} not found in data. Available: {all_prompts}")

    source_prompts = [p for p in all_prompts if p != heldout_prompt]
    if not source_prompts:
        raise ValueError("No source prompts left after removing held-out prompt.")

    print(f"All prompts: {all_prompts}")
    print(f"Held-out prompt: {heldout_prompt}")
    print(f"Source prompts: {source_prompts}")
    for pid in all_prompts:
        print(f"Prompt {pid} active traits: {get_prompt_active_traits(pid, all_traits, score_ranges, prompt_specs)}")

    spec_encoder_name = args.spec_encoder_name or args.encoder_name
    print("Precomputing prompt spec embeddings...")
    prompt_spec_embeddings = precompute_prompt_spec_embeddings(
        prompt_specs=prompt_specs,
        prompt_ids=all_prompts,
        traits=all_traits,
        score_ranges=score_ranges,
        spec_encoder_name=spec_encoder_name,
        max_length=args.max_length,
        device=device,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name, use_fast=True)
    if device.type == 'cuda' and args.amp_dtype == 'bf16' and not torch.cuda.is_bf16_supported():
        print('CUDA bf16 is not supported on this GPU; falling back to full precision.')
        args.amp_dtype = 'none'

    if device.type == 'cuda' and not args.first_order and not args.freeze_encoder:
        print('Warning: second-order meta-training with a full unfrozen encoder is memory-heavy. Consider --first_order or --freeze_encoder if you still hit OOM.')

    model = PromptContextAESModel(
        encoder_name=args.encoder_name,
        traits=all_traits,
        context_dim=args.context_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        context_init_mode=args.context_init_mode,
        freeze_encoder=args.freeze_encoder,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Split source pools once for meta-training.
    source_train_pools: Dict[str, pd.DataFrame] = {}
    source_dev_pools: Dict[str, pd.DataFrame] = {}
    for i, pid in enumerate(source_prompts):
        tr, dv = split_source_prompt_pool(
            df=data_module.df,
            prompt_col=args.prompt_col,
            prompt_id=pid,
            dev_frac=args.source_dev_frac,
            seed=args.seed + i,
        )
        source_train_pools[pid] = tr
        source_dev_pools[pid] = dv
        print(f"Source prompt {pid}: train={len(tr)} dev={len(dv)}")

    # Meta-train on source prompts.
    meta_info = meta_train(
        model=model,
        tokenizer=tokenizer,
        data_module=data_module,
        source_train_pools=source_train_pools,
        source_dev_pools=source_dev_pools,
        prompt_spec_embeddings=prompt_spec_embeddings,
        traits=all_traits,
        score_ranges=score_ranges,
        output_dir=args.output_dir,
        meta_steps=args.meta_steps,
        meta_batch_tasks=args.meta_batch_tasks,
        support_size=args.support_size,
        query_size=args.query_size,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        meta_lr=args.meta_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        device=device,
        val_every=args.val_every,
        val_episodes_per_prompt=args.val_episodes_per_prompt,
        grad_clip=args.grad_clip,
        first_order=args.first_order,
        seed=args.seed,
        amp_dtype=args.amp_dtype,
    )

    print(f"Best meta step: {meta_info['best_step']}")
    print(f"Best source-dev mean QWK: {meta_info['best_val_metrics']['mean_qwk']:.4f}")

    ckpt = torch.load(os.path.join(args.output_dir, 'best_meta_checkpoint.pt'), map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.to(device)
    model.eval()

    results_rows = []
    all_eval_json: Dict[str, Any] = {
        'meta_info': {
            'best_step': meta_info['best_step'],
            'best_val_metrics': meta_info['best_val_metrics'],
        },
        'heldout_prompt': heldout_prompt,
        'results': {},
    }

    for repeat_idx in range(1, args.repeats + 1):
        repeat_name = f"repeat_{repeat_idx:02d}"
        split_seed = args.seed + 1000 * repeat_idx
        held_train, held_dev, held_test = split_heldout_prompt(
            df=data_module.df,
            prompt_col=args.prompt_col,
            heldout_prompt=heldout_prompt,
            dev_frac=args.heldout_dev_frac,
            test_frac=args.heldout_test_frac,
            seed=split_seed,
        )
        print(f"{repeat_name} | held-out split sizes: train={len(held_train)} dev={len(held_dev)} test={len(held_test)}")

        all_eval_json['results'][repeat_name] = {}

        for k in args.k_values:
            if k > len(held_train):
                print(f"Skipping k={k} for {repeat_name}: only {len(held_train)} training essays available.")
                continue

            support_df = held_train.sample(n=k, random_state=split_seed + k).reset_index(drop=True)

            adapt_info = adapt_and_select_phi(
                model=model,
                tokenizer=tokenizer,
                data_module=data_module,
                prompt_id=heldout_prompt,
                support_df=support_df,
                dev_df=held_dev,
                prompt_spec_embedding=prompt_spec_embeddings[heldout_prompt],
                traits=all_traits,
                score_ranges=score_ranges,
                max_length=args.max_length,
                device=device,
                adapt_steps=args.adapt_steps,
                inner_lr=args.inner_lr,
                grad_clip=args.grad_clip,
                amp_dtype=args.amp_dtype,
            )

            zero_phi = model.initial_context(prompt_spec_embeddings[heldout_prompt].to(device)).detach().cpu()
            zero_test_metrics = evaluate_with_phi(
                model=model,
                tokenizer=tokenizer,
                data_module=data_module,
                prompt_id=heldout_prompt,
                eval_df=held_test,
                phi=zero_phi,
                traits=all_traits,
                score_ranges=score_ranges,
                max_length=args.max_length,
                device=device,
                amp_dtype=args.amp_dtype,
            )
            best_test_metrics = evaluate_with_phi(
                model=model,
                tokenizer=tokenizer,
                data_module=data_module,
                prompt_id=heldout_prompt,
                eval_df=held_test,
                phi=adapt_info['best_phi'],
                traits=all_traits,
                score_ranges=score_ranges,
                max_length=args.max_length,
                device=device,
                amp_dtype=args.amp_dtype,
            )

            row = {
                'heldout_prompt': heldout_prompt,
                'repeat_name': repeat_name,
                'fewshot_k': k,
                'train_n': len(support_df),
                'dev_n': len(held_dev),
                'test_n': len(held_test),
                'best_inner_step': adapt_info['best_step'],
                'best_dev_mean_qwk': adapt_info['best_dev_metrics']['mean_qwk'],
                'best_dev_mean_rmse': adapt_info['best_dev_metrics']['mean_rmse'],
                'final_test_mean_qwk': best_test_metrics['mean_qwk'],
                'final_test_mean_rmse': best_test_metrics['mean_rmse'],
                'zero_shot_test_mean_qwk': zero_test_metrics['mean_qwk'],
                'zero_shot_test_mean_rmse': zero_test_metrics['mean_rmse'],
            }
            if 'overall_from_traits' in best_test_metrics:
                row['final_test_overall_qwk'] = best_test_metrics['overall_from_traits']['qwk']
                row['final_test_overall_rmse'] = best_test_metrics['overall_from_traits']['rmse']
                row['zero_shot_test_overall_qwk'] = zero_test_metrics['overall_from_traits']['qwk']
                row['zero_shot_test_overall_rmse'] = zero_test_metrics['overall_from_traits']['rmse']

            row['active_traits'] = ",".join(best_test_metrics.get('active_traits', []))
            for t in all_traits:
                best_tm = best_test_metrics['trait_metrics'].get(t)
                zero_tm = zero_test_metrics['trait_metrics'].get(t)
                row[f'test_{t}_qwk'] = best_tm['qwk'] if best_tm is not None else np.nan
                row[f'test_{t}_rmse'] = best_tm['rmse'] if best_tm is not None else np.nan
                row[f'zero_{t}_qwk'] = zero_tm['qwk'] if zero_tm is not None else np.nan
                row[f'zero_{t}_rmse'] = zero_tm['rmse'] if zero_tm is not None else np.nan

            results_rows.append(row)
            all_eval_json['results'][repeat_name][f'k_{k}'] = {
                'summary_row': row,
                'best_dev_metrics': adapt_info['best_dev_metrics'],
                'dev_metrics_by_step': adapt_info['dev_metrics_by_step'],
                'final_test_metrics': best_test_metrics,
                'zero_shot_test_metrics': zero_test_metrics,
            }

            print(
                f"{repeat_name} k={k} | "
                f"dev_qwk={adapt_info['best_dev_metrics']['mean_qwk']:.4f} | "
                f"test_qwk={best_test_metrics['mean_qwk']:.4f} | "
                f"zero_qwk={zero_test_metrics['mean_qwk']:.4f}"
            )

    results_df = pd.DataFrame(results_rows)
    results_csv = os.path.join(args.output_dir, 'results_summary.csv')
    results_json = os.path.join(args.output_dir, 'results_full.json')
    results_df.to_csv(results_csv, index=False)
    save_json(all_eval_json, results_json)

    if len(results_df) > 0:
        grouped = results_df.groupby('fewshot_k').agg(
            final_test_mean_qwk_mean=('final_test_mean_qwk', 'mean'),
            final_test_mean_qwk_std=('final_test_mean_qwk', 'std'),
            final_test_mean_rmse_mean=('final_test_mean_rmse', 'mean'),
            zero_shot_test_mean_qwk_mean=('zero_shot_test_mean_qwk', 'mean'),
            zero_shot_test_mean_rmse_mean=('zero_shot_test_mean_rmse', 'mean'),
        ).reset_index()
        grouped.to_csv(os.path.join(args.output_dir, 'results_by_k.csv'), index=False)
        print("\nAverage by k:")
        print(grouped.to_string(index=False))

    print(f"\nSaved summary CSV to: {results_csv}")
    print(f"Saved detailed JSON to: {results_json}")


if __name__ == '__main__':
    main()
