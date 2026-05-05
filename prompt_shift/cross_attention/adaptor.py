#!/usr/bin/env python3
"""
Rubric-Conditioned, Partial Fine-Tuning with Cross-Attention for ASAP AES.

This script is written for the user's uploaded ASAP-style files:
- asap_train_with_all_traits.tsv
- asap_prompt_meta.json
- asap_score_ranges.json

What this implementation does:
1. Treats each prompt as a task/domain.
2. Builds structured prompt/rubric text from the prompt metadata JSON.
3. Uses a separate metadata encoder for prompt/rubric encoding.
4. Uses partial full fine-tuning of the essay encoder by unfreezing only the top
   transformer layers while keeping lower layers frozen.
5. Uses cross-attention from essay hidden states to rubric hidden states.
6. Uses episodic first-order meta-learning (FOMAML-style with `higher`) across
   source prompts.
7. Adapts to a held-out prompt with k-shot target training.
8. Reports trait-wise and mean QWK/RMSE on the held-out test set using raw-score prediction. Labels are kept in original prompt-specific score scales.

Important note:
This is a practical implementation of the method. Instead of LoRA, the essay
encoder is partially unfrozen: only the top N transformer blocks are trainable,
while lower layers remain frozen. This lets you do deeper encoder adaptation
without the full memory cost of end-to-end fine-tuning.

Example:
python adaptor_partial_ft.py \
  --train_tsv /mnt/data/asap_train_with_all_traits.tsv \
  --prompt_meta_json /mnt/data/asap_prompt_meta.json \
  --score_ranges_json /mnt/data/asap_score_ranges.json \
  --heldout_prompt 2 \
  --k_shot 32 \
  --output_dir /mnt/data/rclora_results
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

try:
    import higher  # type: ignore
except ImportError as exc:
    raise ImportError(
        "This script requires the `higher` package for FOMAML-style meta-learning. "
        "Install it with: pip install higher"
    ) from exc


# -----------------------------
# Utilities
# -----------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_dump(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# -----------------------------
# Data helpers
# -----------------------------

ALL_TRAITS = [
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


def build_prompt_text(
    prompt_id: str,
    prompt_meta: Dict[str, Any],
    score_ranges: Dict[str, Dict[str, List[float]]],
) -> str:
    meta = prompt_meta[str(prompt_id)]
    trait_sections = []
    trait_guidance = meta.get("trait_guidance", {})

    for trait in meta["traits"]:
        if trait not in ALL_TRAITS:
            continue
        lo, hi = score_ranges[str(prompt_id)][trait]
        section_lines = [f"TRAIT: {trait}", f"RANGE: {lo}-{hi}"]
        guidance = trait_guidance.get(trait, {})
        if guidance.get("definition"):
            section_lines.append(f"DEFINITION: {guidance['definition']}")
        if guidance.get("high_score"):
            section_lines.append(f"HIGH SCORE CUES: {guidance['high_score']}")
        if guidance.get("mid_score"):
            section_lines.append(f"MID SCORE CUES: {guidance['mid_score']}")
        if guidance.get("low_score"):
            section_lines.append(f"LOW SCORE CUES: {guidance['low_score']}")
        trait_sections.append("\n".join(section_lines))

    special_notes = meta.get("special_notes", {})
    note_lines = [f"{k}: {v}" for k, v in special_notes.items()]

    text_parts = [
        f"PROMPT ID: {prompt_id}",
        f"GENRE: {meta.get('genre', 'unknown')}",
        f"PROMPT TYPE: {meta.get('prompt_type', 'unknown')}",
        f"SOURCE DEPENDENT: {meta.get('source_dependent', False)}",
        f"DESCRIPTION: {meta['description']}",
        f"RUBRIC SUMMARY: {meta['rubric']}",
        "TRAIT SCHEMA:",
        "\n\n".join(trait_sections),
    ]
    if note_lines:
        text_parts.extend(["SPECIAL NOTES:", "\n".join(note_lines)])
    return "\n".join(text_parts)


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def get_trait_range(prompt_id: str, trait: str, trait_ranges: Dict[str, Dict[str, Tuple[float, float]]]) -> Tuple[float, float]:
    lo, hi = trait_ranges[str(prompt_id)][trait]
    return float(lo), float(hi)


def clip_score(prompt_id: str, trait: str, value: float, trait_ranges: Dict[str, Dict[str, Tuple[float, float]]]) -> float:
    lo, hi = get_trait_range(prompt_id, trait, trait_ranges)
    return float(max(lo, min(hi, value)))


def clip_round_score(prompt_id: str, trait: str, value: float, trait_ranges: Dict[str, Dict[str, Tuple[float, float]]]) -> float:
    return float(round(clip_score(prompt_id, trait, value, trait_ranges)))


def sample_trait_mean(sample: EssaySample) -> Optional[float]:
    values = [v for v in sample.trait_scores.values() if v is not None]
    if not values:
        return None
    return float(np.mean(values))


def trait_mean_to_bin(score: float) -> int:
    if score <= 1.5:
        return 0
    if score <= 2.5:
        return 1
    if score <= 3.5:
        return 2
    if score <= 4.5:
        return 3
    return 4


def stratified_kshot_split(samples: List[EssaySample], k_shot: int, dev_frac: float, seed: int) -> Tuple[List[EssaySample], List[EssaySample], List[EssaySample]]:
    rng = random.Random(seed)
    idxs = list(range(len(samples)))
    labels = []
    can_stratify = True
    for s in samples:
        mean_score = sample_trait_mean(s)
        if mean_score is None:
            can_stratify = False
            break
        labels.append(trait_mean_to_bin(float(mean_score)))

    if can_stratify and len(set(labels)) > 1:
        try:
            train_idxs, rest_idxs = train_test_split(
                idxs,
                train_size=k_shot,
                random_state=seed,
                stratify=labels,
            )
        except Exception:
            rng.shuffle(idxs)
            train_idxs = idxs[:k_shot]
            rest_idxs = idxs[k_shot:]
    else:
        rng.shuffle(idxs)
        train_idxs = idxs[:k_shot]
        rest_idxs = idxs[k_shot:]

    rest = [samples[i] for i in rest_idxs]
    if not rest:
        return [samples[i] for i in train_idxs], [], []

    dev_size = max(1, int(round(len(rest) * dev_frac)))
    dev_size = min(dev_size, len(rest) - 1) if len(rest) > 1 else len(rest)
    rng.shuffle(rest)
    dev = rest[:dev_size]
    test = rest[dev_size:]
    return [samples[i] for i in train_idxs], dev, test



@dataclass
class EssaySample:
    essay_id: int
    prompt_id: str
    essay: str
    trait_scores: Dict[str, Optional[float]]


class AESDataset(Dataset):
    def __init__(self, samples: List[EssaySample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> EssaySample:
        return self.samples[idx]


class AESCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        prompt_text_by_id: Dict[str, str],
        trait_ranges: Dict[str, Dict[str, Tuple[float, float]]],
        max_essay_length: int,
        max_meta_length: int,
    ):
        self.tokenizer = tokenizer
        self.prompt_text_by_id = prompt_text_by_id
        self.trait_ranges = trait_ranges
        self.max_essay_length = max_essay_length
        self.max_meta_length = max_meta_length
        self.target_names = list(ALL_TRAITS)

    def __call__(self, batch: List[EssaySample]) -> Dict[str, Any]:
        essays = [x.essay for x in batch]
        prompt_ids = [str(x.prompt_id) for x in batch]
        meta_texts = [self.prompt_text_by_id[p] for p in prompt_ids]

        essay_enc = self.tokenizer(
            essays,
            padding=True,
            truncation=True,
            max_length=self.max_essay_length,
            return_tensors="pt",
        )
        meta_enc = self.tokenizer(
            meta_texts,
            padding=True,
            truncation=True,
            max_length=self.max_meta_length,
            return_tensors="pt",
        )

        batch_size = len(batch)
        n_traits = len(self.target_names)
        labels = torch.zeros(batch_size, n_traits, dtype=torch.float32)
        label_mask = torch.zeros(batch_size, n_traits, dtype=torch.float32)
        trait_lows = torch.zeros(batch_size, n_traits, dtype=torch.float32)
        trait_highs = torch.zeros(batch_size, n_traits, dtype=torch.float32)
        trait_spans = torch.zeros(batch_size, n_traits, dtype=torch.float32)
        trait_midpoints = torch.zeros(batch_size, n_traits, dtype=torch.float32)

        for i, sample in enumerate(batch):
            p = str(sample.prompt_id)
            for j, trait in enumerate(self.target_names):
                if trait in self.trait_ranges.get(p, {}):
                    lo, hi = get_trait_range(p, trait, self.trait_ranges)
                    trait_lows[i, j] = lo
                    trait_highs[i, j] = hi
                    trait_spans[i, j] = hi - lo
                    trait_midpoints[i, j] = 0.5 * (lo + hi)

                raw_value = sample.trait_scores.get(trait)
                if raw_value is None:
                    continue
                labels[i, j] = float(raw_value)
                label_mask[i, j] = 1.0

        return {
            "essay_input_ids": essay_enc["input_ids"],
            "essay_attention_mask": essay_enc["attention_mask"],
            "meta_input_ids": meta_enc["input_ids"],
            "meta_attention_mask": meta_enc["attention_mask"],
            "labels": labels,
            "label_mask": label_mask,
            "trait_lows": trait_lows,
            "trait_highs": trait_highs,
            "trait_spans": trait_spans,
            "trait_midpoints": trait_midpoints,
            "prompt_ids": prompt_ids,
            "essay_ids": [x.essay_id for x in batch],
        }


def df_to_samples(df: pd.DataFrame) -> List[EssaySample]:
    samples: List[EssaySample] = []
    for row in df.itertuples(index=False):
        trait_scores = {trait: safe_float(getattr(row, trait, None)) for trait in ALL_TRAITS}
        samples.append(
            EssaySample(
                essay_id=int(row.essay_id),
                prompt_id=str(row.essay_set),
                essay=str(row.essay),
                trait_scores=trait_scores,
            )
        )
    return samples


def group_by_prompt(samples: List[EssaySample]) -> Dict[str, List[EssaySample]]:
    grouped: Dict[str, List[EssaySample]] = {}
    for x in samples:
        grouped.setdefault(str(x.prompt_id), []).append(x)
    return grouped


def random_support_query_split(samples: List[EssaySample], k_support: int, k_query: int, rng: random.Random) -> Tuple[List[EssaySample], List[EssaySample]]:
    assert len(samples) >= (k_support + k_query), "Not enough samples for support/query split"
    idxs = list(range(len(samples)))
    rng.shuffle(idxs)
    support = [samples[i] for i in idxs[:k_support]]
    query = [samples[i] for i in idxs[k_support:k_support + k_query]]
    return support, query

# -----------------------------
# Partial full fine-tuning helpers
# -----------------------------


def get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return model.encoder.layer
    if hasattr(model, "transformer") and hasattr(model.transformer, "layer"):
        return model.transformer.layer
    raise ValueError("Could not find transformer layers on essay encoder. Expected encoder.layer or transformer.layer.")


def freeze_all_parameters(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def set_trainable(module: nn.Module, trainable: bool = True) -> None:
    for p in module.parameters():
        p.requires_grad = trainable


# -----------------------------
# Model
# -----------------------------


class RubricConditionedPartialFTAES(nn.Module):
    def __init__(
        self,
        model_name: str,
        target_names: List[str],
        unfreeze_top_n_layers: int = 2,
        cross_attn_heads: int = 8,
        head_dropout: float = 0.1,
        train_meta_encoder: bool = True,
    ):
        super().__init__()
        self.target_names = target_names

        self.essay_encoder = AutoModel.from_pretrained(model_name)
        self.meta_encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = int(self.essay_encoder.config.hidden_size)
        self.meta_trainable = train_meta_encoder
        self.unfreeze_top_n_layers = max(0, int(unfreeze_top_n_layers))

        freeze_all_parameters(self.essay_encoder)
        self._unfreeze_top_essay_layers(self.unfreeze_top_n_layers)

        if not train_meta_encoder:
            freeze_all_parameters(self.meta_encoder)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=cross_attn_heads,
            dropout=head_dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(self.hidden_size)
        self.cross_dropout = nn.Dropout(head_dropout)

        self.trait_queries = nn.Parameter(torch.randn(len(target_names), self.hidden_size) * 0.02)
        self.trait_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=cross_attn_heads,
            dropout=head_dropout,
            batch_first=True,
        )
        self.trait_norm = nn.LayerNorm(self.hidden_size)

        self.range_feature_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Dropout(head_dropout),
        )

        head_input_dim = self.hidden_size + 32
        self.heads = nn.ModuleDict()
        for name in target_names:
            self.heads[name] = nn.Sequential(
                nn.LayerNorm(head_input_dim),
                nn.Linear(head_input_dim, self.hidden_size),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(self.hidden_size, 1),
            )

    def _unfreeze_top_essay_layers(self, top_n: int) -> None:
        layers = get_transformer_layers(self.essay_encoder)
        if top_n <= 0:
            return
        top_n = min(top_n, len(layers))
        for layer in layers[-top_n:]:
            set_trainable(layer, True)

    @staticmethod
    def masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (hidden * mask).sum(dim=1) / denom

    def forward(
        self,
        essay_input_ids: torch.Tensor,
        essay_attention_mask: torch.Tensor,
        meta_input_ids: torch.Tensor,
        meta_attention_mask: torch.Tensor,
        trait_lows: torch.Tensor,
        trait_highs: torch.Tensor,
        trait_spans: torch.Tensor,
        trait_midpoints: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.meta_trainable:
            meta_outputs = self.meta_encoder(
                input_ids=meta_input_ids,
                attention_mask=meta_attention_mask,
                return_dict=True,
            )
            rubric_hidden = meta_outputs.last_hidden_state
        else:
            with torch.no_grad():
                meta_outputs = self.meta_encoder(
                    input_ids=meta_input_ids,
                    attention_mask=meta_attention_mask,
                    return_dict=True,
                )
                rubric_hidden = meta_outputs.last_hidden_state
        rubric_pooled = self.masked_mean(rubric_hidden, meta_attention_mask)

        essay_outputs = self.essay_encoder(
            input_ids=essay_input_ids,
            attention_mask=essay_attention_mask,
            return_dict=True,
        )
        essay_hidden = essay_outputs.last_hidden_state

        cross_hidden, _ = self.cross_attn(
            query=essay_hidden,
            key=rubric_hidden,
            value=rubric_hidden,
            key_padding_mask=(meta_attention_mask == 0),
            need_weights=False,
        )
        conditioned_hidden = self.cross_norm(essay_hidden + self.cross_dropout(cross_hidden))

        batch_size = conditioned_hidden.size(0)
        trait_queries = self.trait_queries.unsqueeze(0).expand(batch_size, -1, -1)
        trait_queries = trait_queries + rubric_pooled.unsqueeze(1)
        trait_hidden, _ = self.trait_attn(
            query=trait_queries,
            key=conditioned_hidden,
            value=conditioned_hidden,
            key_padding_mask=(essay_attention_mask == 0),
            need_weights=False,
        )
        trait_hidden = self.trait_norm(trait_hidden + trait_queries)

        range_inputs = torch.stack(
            [trait_lows, trait_highs, trait_spans, trait_midpoints],
            dim=-1,
        )
        range_features = self.range_feature_net(range_inputs)

        preds = []
        for idx, name in enumerate(self.target_names):
            head_input = torch.cat([trait_hidden[:, idx, :], range_features[:, idx, :]], dim=-1)
            raw_delta = self.heads[name](head_input).squeeze(-1)
            half_span = 0.5 * trait_spans[:, idx]
            pred = trait_midpoints[:, idx] + half_span * torch.tanh(raw_delta)
            preds.append(pred)
        preds_tensor = torch.stack(preds, dim=1)

        return {
            "preds": preds_tensor,
            "rubric_pooled": rubric_pooled,
        }


# -----------------------------
# Training / eval helpers
# -----------------------------


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out



def compute_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, Any], huber_delta: float = 0.5) -> torch.Tensor:
    preds = outputs["preds"]
    labels = batch["labels"]
    mask = batch["label_mask"]
    trait_spans = batch["trait_spans"].clamp(min=1.0)

    scaled_error = (preds - labels) / trait_spans
    abs_error = scaled_error.abs()
    quadratic = torch.minimum(abs_error, torch.tensor(huber_delta, device=abs_error.device, dtype=abs_error.dtype))
    linear = abs_error - quadratic
    huber = 0.5 * quadratic ** 2 + huber_delta * linear

    denom = mask.sum().clamp(min=1.0)
    return (huber * mask).sum() / denom


def predict_dataset(
    model: nn.Module,
    dataset: AESDataset,
    collator: AESCollator,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    model.eval()
    all_preds = []
    all_labels = []
    all_masks = []
    all_prompt_ids: List[str] = []
    all_essay_ids: List[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(
                essay_input_ids=batch["essay_input_ids"],
                essay_attention_mask=batch["essay_attention_mask"],
                meta_input_ids=batch["meta_input_ids"],
                meta_attention_mask=batch["meta_attention_mask"],
                trait_lows=batch["trait_lows"],
                trait_highs=batch["trait_highs"],
                trait_spans=batch["trait_spans"],
                trait_midpoints=batch["trait_midpoints"],
            )
            all_preds.append(outputs["preds"].cpu())
            all_labels.append(batch["labels"].cpu())
            all_masks.append(batch["label_mask"].cpu())
            all_prompt_ids.extend(batch["prompt_ids"])
            all_essay_ids.extend(batch["essay_ids"])

    return {
        "preds": torch.cat(all_preds, dim=0).numpy(),
        "labels": torch.cat(all_labels, dim=0).numpy(),
        "mask": torch.cat(all_masks, dim=0).numpy(),
        "prompt_ids": all_prompt_ids,
        "essay_ids": all_essay_ids,
    }



def compute_metrics(
    pred_pack: Dict[str, Any],
    target_names: List[str],
    trait_ranges: Dict[str, Dict[str, Tuple[float, float]]],
) -> Dict[str, Any]:
    preds = pred_pack["preds"]
    labels = pred_pack["labels"]
    mask = pred_pack["mask"]
    prompt_ids = pred_pack["prompt_ids"]

    metrics: Dict[str, Any] = {}
    qwk_values = []
    rmse_values = []
    rmse_rounded_values = []

    for t_idx, trait in enumerate(target_names):
        y_true_raw: List[float] = []
        y_pred_raw: List[float] = []
        y_true_round: List[float] = []
        y_pred_round: List[float] = []
        for i in range(len(prompt_ids)):
            if mask[i, t_idx] <= 0:
                continue
            p = str(prompt_ids[i])
            true_raw = float(labels[i, t_idx])
            pred_raw = clip_score(p, trait, float(preds[i, t_idx]), trait_ranges)
            true_int = clip_round_score(p, trait, true_raw, trait_ranges)
            pred_int = clip_round_score(p, trait, pred_raw, trait_ranges)
            y_true_raw.append(true_raw)
            y_pred_raw.append(pred_raw)
            y_true_round.append(true_int)
            y_pred_round.append(pred_int)

        if len(y_true_raw) == 0:
            continue

        try:
            qwk = float(cohen_kappa_score(y_true_round, y_pred_round, weights="quadratic"))
        except Exception:
            qwk = 0.0
        rmse = float(math.sqrt(mean_squared_error(y_true_raw, y_pred_raw)))
        rmse_rounded = float(math.sqrt(mean_squared_error(y_true_round, y_pred_round)))
        metrics[trait] = {
            "n": len(y_true_raw),
            "qwk": qwk,
            "rmse": rmse,
            "rmse_rounded": rmse_rounded,
        }
        qwk_values.append(qwk)
        rmse_values.append(rmse)
        rmse_rounded_values.append(rmse_rounded)

    metrics["mean_trait_qwk"] = float(np.mean(qwk_values)) if qwk_values else 0.0
    metrics["mean_trait_rmse"] = float(np.mean(rmse_values)) if rmse_values else 0.0
    metrics["mean_trait_rmse_rounded"] = float(np.mean(rmse_rounded_values)) if rmse_rounded_values else 0.0
    return metrics


def train_supervised_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    device: torch.device,
    grad_clip: float,
    huber_delta: float,
) -> float:
    model.train()
    running = 0.0
    n_steps = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(
            essay_input_ids=batch["essay_input_ids"],
            essay_attention_mask=batch["essay_attention_mask"],
            meta_input_ids=batch["meta_input_ids"],
            meta_attention_mask=batch["meta_attention_mask"],
            trait_lows=batch["trait_lows"],
            trait_highs=batch["trait_highs"],
            trait_spans=batch["trait_spans"],
            trait_midpoints=batch["trait_midpoints"],
        )
        loss = compute_loss(outputs, batch, huber_delta=huber_delta)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running += float(loss.item())
        n_steps += 1
    return running / max(1, n_steps)



def meta_train(
    model: nn.Module,
    source_grouped: Dict[str, List[EssaySample]],
    collator: AESCollator,
    device: torch.device,
    meta_epochs: int,
    meta_steps_per_epoch: int,
    tasks_per_meta_batch: int,
    support_k: int,
    query_k: int,
    meta_query_chunk_size: int,
    inner_steps: int,
    inner_lr: float,
    outer_lr: float,
    outer_weight_decay: float,
    grad_clip: float,
    huber_delta: float,
    seed: int,
) -> Dict[str, List[float]]:
    rng = random.Random(seed)
    prompt_ids = list(source_grouped.keys())
    outer_optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=outer_lr,
        weight_decay=outer_weight_decay,
    )
    history = {"meta_loss": []}

    for epoch in range(1, meta_epochs + 1):
        model.train()
        epoch_losses = []
        for _ in range(meta_steps_per_epoch):
            chosen = prompt_ids if len(prompt_ids) <= tasks_per_meta_batch else rng.sample(prompt_ids, tasks_per_meta_batch)
            valid_tasks = 0
            task_losses = []
            outer_optimizer.zero_grad()

            for prompt_id in chosen:
                prompt_samples = source_grouped[prompt_id]
                if len(prompt_samples) < (support_k + query_k):
                    continue
                support, query = random_support_query_split(prompt_samples, support_k, query_k, rng)
                support_batch = move_batch_to_device(collator(support), device)

                inner_opt = torch.optim.SGD(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=inner_lr,
                )

                with higher.innerloop_ctx(
                    model,
                    inner_opt,
                    copy_initial_weights=False,
                    track_higher_grads=False,
                ) as (fmodel, diffopt):
                    for _inner in range(inner_steps):
                        s_out = fmodel(
                            essay_input_ids=support_batch["essay_input_ids"],
                            essay_attention_mask=support_batch["essay_attention_mask"],
                            meta_input_ids=support_batch["meta_input_ids"],
                            meta_attention_mask=support_batch["meta_attention_mask"],
                            trait_lows=support_batch["trait_lows"],
                            trait_highs=support_batch["trait_highs"],
                            trait_spans=support_batch["trait_spans"],
                            trait_midpoints=support_batch["trait_midpoints"],
                        )
                        s_loss = compute_loss(s_out, support_batch, huber_delta=huber_delta)
                        diffopt.step(s_loss)
                        del s_out, s_loss

                    chunk_size = max(1, meta_query_chunk_size)
                    query_losses = []
                    for start_idx in range(0, len(query), chunk_size):
                        query_chunk = query[start_idx:start_idx + chunk_size]
                        query_batch = move_batch_to_device(collator(query_chunk), device)
                        q_out = fmodel(
                            essay_input_ids=query_batch["essay_input_ids"],
                            essay_attention_mask=query_batch["essay_attention_mask"],
                            meta_input_ids=query_batch["meta_input_ids"],
                            meta_attention_mask=query_batch["meta_attention_mask"],
                            trait_lows=query_batch["trait_lows"],
                            trait_highs=query_batch["trait_highs"],
                            trait_spans=query_batch["trait_spans"],
                            trait_midpoints=query_batch["trait_midpoints"],
                        )
                        q_loss = compute_loss(q_out, query_batch, huber_delta=huber_delta)
                        query_losses.append(q_loss)
                        del q_out, query_batch

                    if not query_losses:
                        del support_batch
                        continue

                    q_loss = torch.stack(query_losses).mean()
                    scaled_loss = q_loss / max(1, len(chosen))
                    scaled_loss.backward()
                    task_losses.append(float(q_loss.item()))
                    valid_tasks += 1

                    del support_batch, query_losses, q_loss, scaled_loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if valid_tasks == 0:
                continue
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            outer_optimizer.step()
            epoch_losses.append(float(np.mean(task_losses)) if task_losses else 0.0)

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        history["meta_loss"].append(mean_loss)
        print(f"[meta] epoch={epoch:02d} meta_loss={mean_loss:.6f}")

    return history


# -----------------------------
# Main experiment
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tsv", type=str, required=True)
    parser.add_argument("--prompt_meta_json", type=str, required=True)
    parser.add_argument("--score_ranges_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--heldout_prompt", type=str, required=True)
    parser.add_argument("--k_shot", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_essay_length", type=int, default=384)
    parser.add_argument("--max_meta_length", type=int, default=160)

    parser.add_argument("--meta_epochs", type=int, default=5)
    parser.add_argument("--meta_steps_per_epoch", type=int, default=50)
    parser.add_argument("--tasks_per_meta_batch", type=int, default=1)
    parser.add_argument("--support_k", type=int, default=4)
    parser.add_argument("--query_k", type=int, default=4)
    parser.add_argument("--meta_query_chunk_size", type=int, default=4)
    parser.add_argument("--inner_steps", type=int, default=1)
    parser.add_argument("--inner_lr", type=float, default=1e-2)
    parser.add_argument("--outer_lr", type=float, default=2e-4)
    parser.add_argument("--outer_weight_decay", type=float, default=1e-2)

    parser.add_argument("--adapt_epochs", type=int, default=15)
    parser.add_argument("--adapt_batch_size", type=int, default=4)
    parser.add_argument("--adapt_lr", type=float, default=2e-4)
    parser.add_argument("--adapt_weight_decay", type=float, default=1e-2)
    parser.add_argument("--dev_frac", type=float, default=0.5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    parser.add_argument("--unfreeze_top_n_layers", type=int, default=2)
    parser.add_argument("--cross_attn_heads", type=int, default=8)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--huber_delta", type=float, default=0.5)
    parser.add_argument("--freeze_meta_encoder", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    return parser.parse_args()



def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    json_dump(vars(args), os.path.join(args.output_dir, "config.json"))

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"partial_ft_top_layers={args.unfreeze_top_n_layers} freeze_meta_encoder={args.freeze_meta_encoder}")

    df = pd.read_csv(args.train_tsv, sep="\t")
    prompt_meta = load_json(args.prompt_meta_json)
    score_ranges_raw = load_json(args.score_ranges_json)
    score_ranges: Dict[str, Dict[str, Tuple[float, float]]] = {
        str(pid): {trait: (float(r[0]), float(r[1])) for trait, r in trait_map.items()}
        for pid, trait_map in score_ranges_raw.items()
    }

    prompt_text_by_id = {
        str(pid): build_prompt_text(str(pid), prompt_meta, score_ranges_raw)
        for pid in prompt_meta.keys()
    }

    samples = df_to_samples(df)
    grouped = group_by_prompt(samples)
    heldout_prompt = str(args.heldout_prompt)
    assert heldout_prompt in grouped, f"Held-out prompt {heldout_prompt} not found"

    target_names = list(ALL_TRAITS)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collator = AESCollator(
        tokenizer=tokenizer,
        prompt_text_by_id=prompt_text_by_id,
        trait_ranges=score_ranges,
        max_essay_length=args.max_essay_length,
        max_meta_length=args.max_meta_length,
    )

    source_grouped = {p: s for p, s in grouped.items() if p != heldout_prompt}
    heldout_samples = grouped[heldout_prompt]

    aggregate_rows = []
    meta_history_path = os.path.join(args.output_dir, "meta_history.json")

    # Meta-train once and reuse for repeated target adaptations.
    base_model = RubricConditionedPartialFTAES(
        model_name=args.model_name,
        target_names=target_names,
        unfreeze_top_n_layers=args.unfreeze_top_n_layers,
        cross_attn_heads=args.cross_attn_heads,
        head_dropout=args.head_dropout,
        train_meta_encoder=not args.freeze_meta_encoder,
    ).to(device)
    if args.gradient_checkpointing:
        base_model.essay_encoder.gradient_checkpointing_enable()
        if not args.freeze_meta_encoder:
            base_model.meta_encoder.gradient_checkpointing_enable()

    print("Starting episodic meta-training on source prompts...")
    meta_history = meta_train(
        model=base_model,
        source_grouped=source_grouped,
        collator=collator,
        device=device,
        meta_epochs=args.meta_epochs,
        meta_steps_per_epoch=args.meta_steps_per_epoch,
        tasks_per_meta_batch=args.tasks_per_meta_batch,
        support_k=args.support_k,
        query_k=args.query_k,
        meta_query_chunk_size=args.meta_query_chunk_size,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        outer_weight_decay=args.outer_weight_decay,
        grad_clip=args.grad_clip,
        huber_delta=args.huber_delta,
        seed=args.seed,
    )
    json_dump(meta_history, meta_history_path)
    meta_state = copy.deepcopy(base_model.state_dict())

    # Optional zero-shot evaluation before adaptation.
    zero_pack = predict_dataset(
        model=base_model,
        dataset=AESDataset(heldout_samples),
        collator=collator,
        batch_size=args.adapt_batch_size,
        device=device,
    )
    zero_metrics = compute_metrics(zero_pack, target_names=target_names, trait_ranges=score_ranges)
    json_dump(zero_metrics, os.path.join(args.output_dir, f"heldout_{heldout_prompt}_zero_shot_metrics.json"))

    for repeat_idx in range(args.repeats):
        repeat_seed = args.seed + repeat_idx
        seed_everything(repeat_seed)
        repeat_dir = os.path.join(args.output_dir, f"repeat_{repeat_idx + 1:02d}")
        ensure_dir(repeat_dir)

        train_samples, dev_samples, test_samples = stratified_kshot_split(
            heldout_samples,
            k_shot=args.k_shot,
            dev_frac=args.dev_frac,
            seed=repeat_seed,
        )

        json_dump(
            {
                "heldout_prompt": heldout_prompt,
                "repeat": repeat_idx + 1,
                "k_shot": args.k_shot,
                "train_n": len(train_samples),
                "dev_n": len(dev_samples),
                "test_n": len(test_samples),
            },
            os.path.join(repeat_dir, "split_info.json"),
        )

        model = RubricConditionedPartialFTAES(
            model_name=args.model_name,
            target_names=target_names,
            unfreeze_top_n_layers=args.unfreeze_top_n_layers,
            cross_attn_heads=args.cross_attn_heads,
            head_dropout=args.head_dropout,
            train_meta_encoder=not args.freeze_meta_encoder,
        ).to(device)
        if args.gradient_checkpointing:
            model.essay_encoder.gradient_checkpointing_enable()
            if not args.freeze_meta_encoder:
                model.meta_encoder.gradient_checkpointing_enable()
        model.load_state_dict(meta_state, strict=True)

        train_loader = DataLoader(
            AESDataset(train_samples),
            batch_size=min(args.adapt_batch_size, max(1, len(train_samples))),
            shuffle=True,
            collate_fn=collator,
        )

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.adapt_lr,
            weight_decay=args.adapt_weight_decay,
        )
        total_steps = max(1, len(train_loader) * args.adapt_epochs)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        best_dev = -1e9
        best_state = None
        history_rows = []

        for epoch in range(1, args.adapt_epochs + 1):
            train_loss = train_supervised_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                grad_clip=args.grad_clip,
                huber_delta=args.huber_delta,
            )

            dev_metrics = None
            if len(dev_samples) > 0:
                dev_pack = predict_dataset(
                    model=model,
                    dataset=AESDataset(dev_samples),
                    collator=collator,
                    batch_size=args.adapt_batch_size,
                    device=device,
                )
                dev_metrics = compute_metrics(dev_pack, target_names=target_names, trait_ranges=score_ranges)
                monitor = dev_metrics["mean_trait_qwk"]
            else:
                monitor = -train_loss

            history_row = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "dev_mean_trait_qwk": float(dev_metrics["mean_trait_qwk"]) if dev_metrics else None,
                "dev_mean_trait_rmse": float(dev_metrics["mean_trait_rmse"]) if dev_metrics else None,
            }
            history_rows.append(history_row)
            print(
                f"[adapt] repeat={repeat_idx+1:02d} epoch={epoch:02d} "
                f"train_loss={train_loss:.6f} "
                f"dev_qwk={monitor:.6f}"
            )

            if monitor > best_dev:
                best_dev = monitor
                best_state = copy.deepcopy(model.state_dict())

        json_dump(history_rows, os.path.join(repeat_dir, "adapt_history.json"))
        if best_state is None:
            best_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state, strict=True)
        torch.save(best_state, os.path.join(repeat_dir, "best_model.pt"))

        dev_metrics = None
        if len(dev_samples) > 0:
            dev_pack = predict_dataset(
                model=model,
                dataset=AESDataset(dev_samples),
                collator=collator,
                batch_size=args.adapt_batch_size,
                device=device,
            )
            dev_metrics = compute_metrics(dev_pack, target_names=target_names, trait_ranges=score_ranges)
            json_dump(dev_metrics, os.path.join(repeat_dir, "dev_metrics.json"))

        test_pack = predict_dataset(
            model=model,
            dataset=AESDataset(test_samples),
            collator=collator,
            batch_size=args.adapt_batch_size,
            device=device,
        )
        test_metrics = compute_metrics(test_pack, target_names=target_names, trait_ranges=score_ranges)
        json_dump(test_metrics, os.path.join(repeat_dir, "test_metrics.json"))

        row = {
            "heldout_prompt": heldout_prompt,
            "repeat": repeat_idx + 1,
            "k_shot": args.k_shot,
            "train_n": len(train_samples),
            "dev_n": len(dev_samples),
            "test_n": len(test_samples),
            "best_dev_mean_trait_qwk": float(dev_metrics["mean_trait_qwk"]) if dev_metrics else None,
            "best_dev_mean_trait_rmse": float(dev_metrics["mean_trait_rmse"]) if dev_metrics else None,
            "test_mean_trait_qwk": float(test_metrics["mean_trait_qwk"]),
            "test_mean_trait_rmse": float(test_metrics["mean_trait_rmse"]),
            "zero_shot_mean_trait_qwk": float(zero_metrics["mean_trait_qwk"]),
            "zero_shot_mean_trait_rmse": float(zero_metrics["mean_trait_rmse"]),
        }
        aggregate_rows.append(row)

    agg_df = pd.DataFrame(aggregate_rows)
    agg_path = os.path.join(args.output_dir, f"heldout_{heldout_prompt}_k_{args.k_shot}_summary.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"Saved summary to {agg_path}")


if __name__ == "__main__":
    main()
