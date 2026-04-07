from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, T5EncoderModel, get_linear_schedule_with_warmup

try:
    from peft import LoraConfig, TaskType, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

EPS = 1e-8
CANONICAL_TRAITS = [
    "content",
    "organization",
    "word_choice",
    "sentence_fluency",
    "conventions",
]
CANON_PREFIX = "__canon__"
MASK_PREFIX = "__mask__"
OVERALL_COL = f"{CANON_PREFIX}overall"
OVERALL_MASK_COL = f"{MASK_PREFIX}overall"

# Fill these with your real essay-set / prompt descriptions if you want them
# embedded automatically when the CSV does not already include a prompt text column.
DEFAULT_SOURCE_PROMPT_MAP: Dict[str, str] = {
    1: "Write a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you.",
    2: "Write a persuasive essay to a newspaper reflecting your views on censorship in libraries. Support your position with convincing arguments from your own experience, observations, and/or reading.",
    3: "Write a response that explains how the features of the setting affect the cyclist. In your response, include examples from the source material that support your conclusion.",
    4: "Write a response that explains why the author concludes the story with the last paragraph of the source material given. In your response, include details and examples from the story that support your ideas.",
    5: "Describe the mood created by the author in the memoir given in the source material. Support your answer with relevant and specific information from the memoir.",
    6: "Based on the excerpt given in the source material, describe the obstacles the builders of the Empire State Building faced in attempting to allow dirigibles to dock there. Support your answer with relevant and specific information from the excerpt.",
    7: "Do only one of the following: write a story about a time when you were patient OR write a story about a time when someone you know was patient OR write a story in your own way about patience.",
    8: "We all understand the benefits of laughter. For example, someone once said, 'Laughter is the shortest distance between two people.' Many other people believe that laughter is an important part of any relationship. Tell a true story in which laughter was one element or part.",
}
DEFAULT_TARGET_PROMPT_MAP: Dict[str, str] = {
    "VUW1": "Should 8-12 year old children use mobile phones?",
    "VUW2": "Should cigarette manufacturers compensate people who develop cancer from cigarette smoking?",
}


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def quadratic_weighted_kappa(y_true: List[int], y_pred: List[int]) -> float:
    if len(y_true) == 0:
        return 0.0
    if len(set(y_true)) < 2 and len(set(y_pred)) < 2:
        return 1.0
    try:
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except Exception:
        return 0.0


def safe_json_load(spec: Optional[str]) -> Optional[dict]:
    if spec is None:
        return None
    spec = spec.strip()
    if not spec:
        return None
    if os.path.exists(spec):
        with open(spec, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(spec)


def normalize_prompt_map_keys(prompt_map: Optional[Dict]) -> Dict[str, str]:
    if not prompt_map:
        return {}
    return {str(k): str(v) for k, v in prompt_map.items()}


@dataclass
class TraitSchema:
    dataset_to_canonical: Dict[str, str]
    canonical_traits: List[str]

    def __post_init__(self) -> None:
        bad_targets = [v for v in self.dataset_to_canonical.values() if v not in self.canonical_traits]
        if bad_targets:
            raise ValueError(
                f"Trait map contains canonical traits outside the allowed set: {sorted(set(bad_targets))}. "
                f"Allowed: {self.canonical_traits}"
            )
        inverse: Dict[str, str] = {}
        for dataset_col, canonical in self.dataset_to_canonical.items():
            if canonical in inverse:
                raise ValueError(
                    f"Multiple dataset columns map to the same canonical trait '{canonical}': "
                    f"'{inverse[canonical]}' and '{dataset_col}'."
                )
            inverse[canonical] = dataset_col
        self.canonical_to_dataset = inverse

    def label_for(self, canonical_trait: str) -> str:
        return self.canonical_to_dataset.get(canonical_trait, canonical_trait)


@dataclass
class PromptNormStats:
    mins: Dict[str, float]
    maxs: Dict[str, float]


class ScoreNormalizer:
    """
    Per-prompt min-max normalization over canonical traits and overall score.
    Missing traits are allowed and ignored.
    """

    def __init__(self, canonical_traits: List[str]):
        self.canonical_traits = canonical_traits
        self.stats: Dict[str, PromptNormStats] = {}

    def fit(self, df: pd.DataFrame, prompt_col: str) -> None:
        self.stats = {}
        for prompt_id, group in df.groupby(prompt_col):
            self.stats[str(prompt_id)] = self._compute_prompt_stats(group)

    def register_single_prompt(self, prompt_id: str, df: pd.DataFrame) -> None:
        self.stats[str(prompt_id)] = self._compute_prompt_stats(df)

    def _compute_prompt_stats(self, df: pd.DataFrame) -> PromptNormStats:
        mins: Dict[str, float] = {}
        maxs: Dict[str, float] = {}
        for canonical in self.canonical_traits:
            canon_col = canon_trait_col(canonical)
            mask_col = mask_trait_col(canonical)
            values = df.loc[df[mask_col] > 0, canon_col].astype(float)
            if len(values) == 0:
                mins[canon_col] = 0.0
                maxs[canon_col] = 1.0
            else:
                mins[canon_col] = float(values.min())
                maxs[canon_col] = float(values.max())
                if abs(maxs[canon_col] - mins[canon_col]) < EPS:
                    maxs[canon_col] = mins[canon_col] + 1.0
        overall_values = df.loc[df[OVERALL_MASK_COL] > 0, OVERALL_COL].astype(float)
        if len(overall_values) == 0:
            mins[OVERALL_COL] = 0.0
            maxs[OVERALL_COL] = 1.0
        else:
            mins[OVERALL_COL] = float(overall_values.min())
            maxs[OVERALL_COL] = float(overall_values.max())
            if abs(maxs[OVERALL_COL] - mins[OVERALL_COL]) < EPS:
                maxs[OVERALL_COL] = mins[OVERALL_COL] + 1.0
        return PromptNormStats(mins=mins, maxs=maxs)

    def normalize_value(self, prompt_id: str, canon_col: str, value: float) -> float:
        stats = self.stats[str(prompt_id)]
        return (float(value) - stats.mins[canon_col]) / (stats.maxs[canon_col] - stats.mins[canon_col] + EPS)

    def denormalize_value(self, prompt_id: str, canon_col: str, value: float) -> float:
        stats = self.stats[str(prompt_id)]
        return stats.mins[canon_col] + float(value) * (stats.maxs[canon_col] - stats.mins[canon_col])

    def clip_round(self, prompt_id: str, canon_col: str, value: float) -> int:
        stats = self.stats[str(prompt_id)]
        value = max(stats.mins[canon_col], min(stats.maxs[canon_col], value))
        return int(round(value))

    def save(self, path: str) -> None:
        payload = {k: asdict(v) for k, v in self.stats.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def canon_trait_col(canonical_trait: str) -> str:
    return f"{CANON_PREFIX}{canonical_trait}"


def mask_trait_col(canonical_trait: str) -> str:
    return f"{MASK_PREFIX}{canonical_trait}"


def default_identity_trait_map(df: pd.DataFrame, canonical_traits: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for trait in canonical_traits:
        if trait in df.columns:
            mapping[trait] = trait
    return mapping


def build_trait_schema(df: pd.DataFrame, trait_map_spec: Optional[str], canonical_traits: List[str]) -> TraitSchema:
    mapping = safe_json_load(trait_map_spec)
    if mapping is None:
        mapping = default_identity_trait_map(df, canonical_traits)
    if not mapping:
        raise ValueError(
            "No usable trait mapping was found. Provide --source_trait_map / --target_trait_map or make your "
            "columns match the canonical trait names."
        )
    return TraitSchema(dataset_to_canonical=mapping, canonical_traits=canonical_traits)


def materialize_canonical_columns(
    df: pd.DataFrame,
    schema: TraitSchema,
    prompt_col: str,
    overall_col: str,
) -> pd.DataFrame:
    out = df.copy()
    if prompt_col not in out.columns:
        raise ValueError(f"Missing prompt column '{prompt_col}'.")

    for canonical in schema.canonical_traits:
        out[canon_trait_col(canonical)] = np.nan
        out[mask_trait_col(canonical)] = 0.0

    for dataset_col, canonical in schema.dataset_to_canonical.items():
        if dataset_col not in out.columns:
            raise ValueError(
                f"Trait map expects dataset column '{dataset_col}', but it was not found in the dataframe."
            )
        out[canon_trait_col(canonical)] = pd.to_numeric(out[dataset_col], errors="coerce")
        out[mask_trait_col(canonical)] = out[canon_trait_col(canonical)].notna().astype(float)

    if overall_col in out.columns:
        out[OVERALL_COL] = pd.to_numeric(out[overall_col], errors="coerce")
        out[OVERALL_MASK_COL] = out[OVERALL_COL].notna().astype(float)
    else:
        canon_cols = [canon_trait_col(t) for t in schema.canonical_traits]
        trait_means = out[canon_cols].mean(axis=1, skipna=True)
        out[OVERALL_COL] = trait_means.round()
        out[OVERALL_MASK_COL] = trait_means.notna().astype(float)

    if float(out[OVERALL_MASK_COL].sum()) == 0.0:
        raise ValueError(
            "Could not construct overall scores. Provide an overall column or enough trait columns to derive one."
        )

    return out


def resolve_prompt_text(
    row: pd.Series,
    prompt_col: str,
    prompt_text_col: Optional[str],
    prompt_map: Optional[Dict[str, str]],
) -> Optional[str]:
    if prompt_text_col and prompt_text_col in row and pd.notna(row[prompt_text_col]):
        value = str(row[prompt_text_col]).strip()
        if value:
            return value
    if prompt_map:
        return prompt_map.get(str(row[prompt_col]))
    return None


def build_text(
    row: pd.Series,
    essay_col: str,
    prompt_col: str,
    prompt_text_col: Optional[str],
    prompt_map: Optional[Dict[str, str]] = None,
) -> str:
    prompt_text = resolve_prompt_text(row, prompt_col, prompt_text_col, prompt_map)
    if prompt_text:
        prompt_prefix = f"Prompt: {prompt_text}\n"
    else:
        prompt_prefix = f"Prompt ID: {row[prompt_col]}\n"
    return prompt_prefix + f"Essay:\n{row[essay_col]}"


def stratify_labels(df: pd.DataFrame, overall_col: str = OVERALL_COL) -> pd.Series:
    values = df[overall_col].astype(float)
    try:
        bins = pd.qcut(values, q=min(3, values.nunique()), labels=False, duplicates="drop")
        if bins.nunique() > 1:
            return bins.astype(int)
    except Exception:
        pass
    return pd.cut(values, bins=3, labels=False, include_lowest=True).astype(int)


# -----------------------------
# Dataset
# -----------------------------

class AESDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        prompt_col: str,
        essay_col: str,
        prompt_text_col: Optional[str],
        normalizer: ScoreNormalizer,
        canonical_traits: List[str],
        max_length: int = 512,
        prompt_map: Optional[Dict[str, str]] = None,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.prompt_col = prompt_col
        self.essay_col = essay_col
        self.prompt_text_col = prompt_text_col
        self.normalizer = normalizer
        self.canonical_traits = canonical_traits
        self.max_length = max_length
        self.prompt_map = normalize_prompt_map_keys(prompt_map)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        prompt_id = str(row[self.prompt_col])
        text = build_text(row, self.essay_col, self.prompt_col, self.prompt_text_col, self.prompt_map)
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        traits: List[float] = []
        trait_mask: List[float] = []
        raw_traits: List[float] = []
        for canonical in self.canonical_traits:
            value = row[canon_trait_col(canonical)]
            mask = float(row[mask_trait_col(canonical)])
            if mask > 0:
                traits.append(self.normalizer.normalize_value(prompt_id, canon_trait_col(canonical), float(value)))
                raw_traits.append(float(value))
                trait_mask.append(1.0)
            else:
                traits.append(0.0)
                raw_traits.append(0.0)
                trait_mask.append(0.0)

        overall_mask = float(row[OVERALL_MASK_COL])
        if overall_mask > 0:
            overall = self.normalizer.normalize_value(prompt_id, OVERALL_COL, float(row[OVERALL_COL]))
            raw_overall = float(row[OVERALL_COL])
        else:
            overall = 0.0
            raw_overall = 0.0

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "traits": torch.tensor(traits, dtype=torch.float),
            "trait_mask": torch.tensor(trait_mask, dtype=torch.float),
            "overall": torch.tensor(overall, dtype=torch.float),
            "overall_mask": torch.tensor(overall_mask, dtype=torch.float),
            "raw_traits": torch.tensor(raw_traits, dtype=torch.float),
            "raw_overall": torch.tensor(raw_overall, dtype=torch.float),
            "prompt_id": prompt_id,
        }


def collate_batch(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in items]),
        "attention_mask": torch.stack([x["attention_mask"] for x in items]),
        "traits": torch.stack([x["traits"] for x in items]),
        "trait_mask": torch.stack([x["trait_mask"] for x in items]),
        "overall": torch.stack([x["overall"] for x in items]),
        "overall_mask": torch.stack([x["overall_mask"] for x in items]),
        "raw_traits": torch.stack([x["raw_traits"] for x in items]),
        "raw_overall": torch.stack([x["raw_overall"] for x in items]),
        "prompt_id": [x["prompt_id"] for x in items],
    }


# -----------------------------
# Model
# -----------------------------

class T5EncoderAESModel(nn.Module):
    def __init__(self, model_name: str, num_traits: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        hidden = self.encoder.config.d_model
        self.dropout = nn.Dropout(dropout)
        self.trait_head = nn.Linear(hidden, num_traits)
        self.overall_head = nn.Linear(hidden, 1)

    def mean_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        masked = hidden_states * mask
        pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(outputs.last_hidden_state, attention_mask)
        return self.dropout(pooled)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        fast_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        features = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        return self.forward_from_features(features, fast_weights)

    def forward_from_features(
        self,
        features: torch.Tensor,
        fast_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if fast_weights is None:
            trait_logits = self.trait_head(features)
            overall_logits = self.overall_head(features).squeeze(-1)
        else:
            trait_logits = F.linear(
                features,
                fast_weights["trait_head.weight"],
                fast_weights["trait_head.bias"],
            )
            overall_logits = F.linear(
                features,
                fast_weights["overall_head.weight"],
                fast_weights["overall_head.bias"],
            ).squeeze(-1)
        return {
            "trait_scores": torch.sigmoid(trait_logits),
            "overall_score": torch.sigmoid(overall_logits),
            "features": features,
        }

    def clone_head_weights(self) -> Dict[str, torch.Tensor]:
        return {
            "trait_head.weight": self.trait_head.weight.clone(),
            "trait_head.bias": self.trait_head.bias.clone(),
            "overall_head.weight": self.overall_head.weight.clone(),
            "overall_head.bias": self.overall_head.bias.clone(),
        }


def freeze_bottom_encoder_layers(model: T5EncoderAESModel, n_trainable_blocks: int = 2) -> None:
    blocks = model.encoder.encoder.block
    total = len(blocks)
    cutoff = max(0, total - n_trainable_blocks)
    for idx, block in enumerate(blocks):
        requires_grad = idx >= cutoff
        for p in block.parameters():
            p.requires_grad = requires_grad
    for p in model.encoder.shared.parameters():
        p.requires_grad = False
    for p in model.encoder.encoder.final_layer_norm.parameters():
        p.requires_grad = True
    for p in model.trait_head.parameters():
        p.requires_grad = True
    for p in model.overall_head.parameters():
        p.requires_grad = True


def apply_lora_to_encoder(model: T5EncoderAESModel, r: int = 8, alpha: int = 16, dropout: float = 0.1) -> T5EncoderAESModel:
    if not PEFT_AVAILABLE:
        raise ImportError("peft is not installed. Install with: pip install -U peft")
    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["q", "v"],
    )
    model.encoder = get_peft_model(model.encoder, config)
    return model


# -----------------------------
# Loss and metrics
# -----------------------------

def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    denom = mask.sum(dim=dim).clamp(min=1.0)
    return (values * mask).sum(dim=dim) / denom


def masked_huber_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 0.1) -> torch.Tensor:
    loss = F.huber_loss(pred, target, reduction="none", delta=delta)
    weighted = loss * mask
    return weighted.sum() / mask.sum().clamp(min=1.0)


def compute_joint_loss(
    pred_traits: torch.Tensor,
    pred_overall: torch.Tensor,
    gold_traits: torch.Tensor,
    trait_mask: torch.Tensor,
    gold_overall: torch.Tensor,
    overall_mask: torch.Tensor,
    consistency_weight: float = 0.25,
    overall_weight: float = 1.0,
) -> torch.Tensor:
    trait_loss = masked_huber_loss(pred_traits, gold_traits, trait_mask, delta=0.1)

    overall_loss = masked_huber_loss(
        pred_overall,
        gold_overall,
        overall_mask,
        delta=0.1,
    )

    valid_consistency_mask = ((trait_mask.sum(dim=1) > 0).float() * overall_mask).float()
    pred_trait_mean = masked_mean(pred_traits, trait_mask, dim=1)
    consistency_terms = F.mse_loss(pred_overall, pred_trait_mean, reduction="none")
    consistency = (consistency_terms * valid_consistency_mask).sum() / valid_consistency_mask.sum().clamp(min=1.0)

    return trait_loss + overall_weight * overall_loss + consistency_weight * consistency


def compute_metrics(
    df: pd.DataFrame,
    pred_traits: np.ndarray,
    pred_overall: np.ndarray,
    normalizer: ScoreNormalizer,
    prompt_col: str,
    schema: TraitSchema,
    canonical_traits: List[str],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    trait_qwks: List[float] = []

    # Align df rows with prediction rows
    df = df.reset_index(drop=True)

    for trait_idx, canonical in enumerate(canonical_traits):
        canon_col = canon_trait_col(canonical)
        mask_col = mask_trait_col(canonical)
        label = schema.label_for(canonical)

        y_true: List[int] = []
        y_pred: List[int] = []

        for i, row in df.iterrows():
            if float(row[mask_col]) <= 0:
                continue

            prompt_id = str(row[prompt_col])
            pred_raw = normalizer.denormalize_value(
                prompt_id, canon_col, float(pred_traits[i, trait_idx])
            )
            y_true.append(int(row[canon_col]))
            y_pred.append(normalizer.clip_round(prompt_id, canon_col, pred_raw))

        if len(y_true) == 0:
            continue

        qwk = quadratic_weighted_kappa(y_true, y_pred)
        metrics[f"qwk_{label}"] = qwk
        trait_qwks.append(qwk)

    metrics["mean_trait_qwk"] = float(np.mean(trait_qwks)) if trait_qwks else 0.0

    overall_true: List[int] = []
    overall_pred_from_traits: List[int] = []
    overall_pred_from_head: List[int] = []
    overall_pred_from_head_cont: List[float] = []

    for i, row in df.iterrows():
        if float(row[OVERALL_MASK_COL]) <= 0:
            continue

        prompt_id = str(row[prompt_col])
        available_trait_raws: List[float] = []

        for trait_idx, canonical in enumerate(canonical_traits):
            if float(row[mask_trait_col(canonical)]) <= 0:
                continue
            canon_col = canon_trait_col(canonical)
            available_trait_raws.append(
                normalizer.denormalize_value(
                    prompt_id, canon_col, float(pred_traits[i, trait_idx])
                )
            )

        if available_trait_raws:
            overall_true.append(int(row[OVERALL_COL]))
            overall_pred_from_traits.append(int(round(float(np.mean(available_trait_raws)))))

            head_raw = normalizer.denormalize_value(
                prompt_id, OVERALL_COL, float(pred_overall[i])
            )
            overall_pred_from_head_cont.append(head_raw)
            overall_pred_from_head.append(
                normalizer.clip_round(prompt_id, OVERALL_COL, head_raw)
            )

    metrics["overall_qwk_from_traits"] = (
        quadratic_weighted_kappa(overall_true, overall_pred_from_traits)
        if overall_true else 0.0
    )
    metrics["overall_qwk_from_head"] = (
        quadratic_weighted_kappa(overall_true, overall_pred_from_head)
        if overall_true else 0.0
    )
    metrics["overall_rmse_from_head"] = (
        float(math.sqrt(mean_squared_error(overall_true, overall_pred_from_head_cont)))
        if overall_true else 0.0
    )

    return metrics


# -----------------------------
# Training helpers
# -----------------------------

def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith("bias") or "layer_norm" in n.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )


@torch.no_grad()
def evaluate_loss(
    model: T5EncoderAESModel,
    loader: DataLoader,
    device: torch.device,
    consistency_weight: float,
    overall_weight: float,
) -> float:
    model.eval()
    running = 0.0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch["input_ids"], batch["attention_mask"])
        loss = compute_joint_loss(
            out["trait_scores"],
            out["overall_score"],
            batch["traits"],
            batch["trait_mask"],
            batch["overall"],
            batch["overall_mask"],
            consistency_weight=consistency_weight,
            overall_weight=overall_weight,
        )
        running += loss.item()
    return running / max(1, len(loader))


def supervised_train(
    model: T5EncoderAESModel,
    train_loader: DataLoader,
    dev_loader: Optional[DataLoader],
    device: torch.device,
    lr: float,
    epochs: int,
    weight_decay: float,
    consistency_weight: float,
    overall_weight: float,
    save_path: str,
) -> None:
    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )
    best_loss = float("inf")
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            out = model(batch["input_ids"], batch["attention_mask"])
            loss = compute_joint_loss(
                out["trait_scores"],
                out["overall_score"],
                batch["traits"],
                batch["trait_mask"],
                batch["overall"],
                batch["overall_mask"],
                consistency_weight=consistency_weight,
                overall_weight=overall_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()

        avg_train = running / max(1, len(train_loader))
        msg = f"[warmup] epoch={epoch} train_loss={avg_train:.4f}"

        if dev_loader is not None:
            dev_loss = evaluate_loss(model, dev_loader, device, consistency_weight, overall_weight)
            msg += f" dev_loss={dev_loss:.4f}"
            if dev_loss < best_loss:
                best_loss = dev_loss
                torch.save(model.state_dict(), save_path)
        else:
            if avg_train < best_loss:
                best_loss = avg_train
                torch.save(model.state_dict(), save_path)

        print(msg)

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))


class EpisodeSampler:
    def __init__(self, df: pd.DataFrame, prompt_col: str, seed: int = 42):
        self.df = df.copy()
        self.prompt_col = prompt_col
        self.rng = random.Random(seed)
        self.groups = {str(k): v.copy() for k, v in self.df.groupby(prompt_col)}

    def sample_episode(self, support_size: int, query_size: int) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
        prompt_id = self.rng.choice(list(self.groups.keys()))
        group = self.groups[prompt_id].copy()
        if len(group) < support_size + query_size:
            raise ValueError(f"Prompt {prompt_id} does not have enough samples for the requested episode size.")
        labels = stratify_labels(group, OVERALL_COL)
        support_idx, remainder_idx = train_test_split(
            np.arange(len(group)),
            train_size=support_size,
            stratify=labels,
            random_state=self.rng.randint(0, 10**6),
        )
        remainder = group.iloc[remainder_idx].copy()
        rem_labels = stratify_labels(remainder, OVERALL_COL)
        if len(remainder) <= query_size:
            query_df = remainder.copy()
        else:
            query_idx, _ = train_test_split(
                np.arange(len(remainder)),
                train_size=query_size,
                stratify=rem_labels,
                random_state=self.rng.randint(0, 10**6),
            )
            query_df = remainder.iloc[query_idx].copy()
        support_df = group.iloc[support_idx].copy()
        return prompt_id, support_df, query_df


def single_batch_from_df(
    df: pd.DataFrame,
    tokenizer,
    prompt_col: str,
    essay_col: str,
    prompt_text_col: Optional[str],
    normalizer: ScoreNormalizer,
    canonical_traits: List[str],
    max_length: int,
    prompt_map: Optional[Dict[str, str]] = None,
) -> Dict[str, torch.Tensor]:
    ds = AESDataset(
        df=df,
        tokenizer=tokenizer,
        prompt_col=prompt_col,
        essay_col=essay_col,
        prompt_text_col=prompt_text_col,
        normalizer=normalizer,
        canonical_traits=canonical_traits,
        max_length=max_length,
        prompt_map=prompt_map,
    )
    items = [ds[i] for i in range(len(ds))]
    return collate_batch(items)


def fast_adapt_head(
    model: T5EncoderAESModel,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    inner_lr: float,
    inner_steps: int,
    consistency_weight: float,
    overall_weight: float,
) -> Dict[str, torch.Tensor]:
    batch = move_batch_to_device(batch, device)
    features = model.encode(batch["input_ids"], batch["attention_mask"])
    fast_weights = model.clone_head_weights()
    for key in fast_weights:
        fast_weights[key] = fast_weights[key].detach().clone().requires_grad_(True)

    for _ in range(inner_steps):
        out = model.forward_from_features(features, fast_weights=fast_weights)
        loss = compute_joint_loss(
            out["trait_scores"],
            out["overall_score"],
            batch["traits"],
            batch["trait_mask"],
            batch["overall"],
            batch["overall_mask"],
            consistency_weight=consistency_weight,
            overall_weight=overall_weight,
        )
        grads = torch.autograd.grad(loss, list(fast_weights.values()), create_graph=False)
        fast_weights = {
            name: (param - inner_lr * grad).detach().clone().requires_grad_(True)
            for (name, param), grad in zip(fast_weights.items(), grads)
        }
    return fast_weights


def meta_train_anil(
    model: T5EncoderAESModel,
    train_df: pd.DataFrame,
    tokenizer,
    normalizer: ScoreNormalizer,
    prompt_col: str,
    essay_col: str,
    prompt_text_col: Optional[str],
    canonical_traits: List[str],
    device: torch.device,
    output_dir: str,
    outer_lr: float,
    weight_decay: float,
    meta_steps: int,
    meta_batch_size: int,
    support_size: int,
    query_size: int,
    inner_lr: float,
    inner_steps: int,
    consistency_weight: float,
    overall_weight: float,
    max_length: int,
    prompt_map: Optional[Dict[str, str]] = None,
) -> None:
    optimizer = build_optimizer(model, lr=outer_lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * meta_steps)),
        num_training_steps=max(1, meta_steps),
    )
    sampler = EpisodeSampler(train_df, prompt_col=prompt_col)
    best_meta_loss = float("inf")
    save_path = os.path.join(output_dir, "best_meta_model.pt")
    model.to(device)

    for step in range(1, meta_steps + 1):
        model.train()
        meta_loss = 0.0
        optimizer.zero_grad()

        for _ in range(meta_batch_size):
            _, support_df, query_df = sampler.sample_episode(support_size=support_size, query_size=query_size)
            support_batch = single_batch_from_df(
                support_df,
                tokenizer,
                prompt_col,
                essay_col,
                prompt_text_col,
                normalizer,
                canonical_traits,
                max_length,
                prompt_map=prompt_map,
            )
            query_batch = single_batch_from_df(
                query_df,
                tokenizer,
                prompt_col,
                essay_col,
                prompt_text_col,
                normalizer,
                canonical_traits,
                max_length,
                prompt_map=prompt_map,
            )
            fast_weights = fast_adapt_head(
                model,
                support_batch,
                device,
                inner_lr=inner_lr,
                inner_steps=inner_steps,
                consistency_weight=consistency_weight,
                overall_weight=overall_weight,
            )
            query_batch = move_batch_to_device(query_batch, device)
            out = model(query_batch["input_ids"], query_batch["attention_mask"], fast_weights=fast_weights)
            loss = compute_joint_loss(
                out["trait_scores"],
                out["overall_score"],
                query_batch["traits"],
                query_batch["trait_mask"],
                query_batch["overall"],
                query_batch["overall_mask"],
                consistency_weight=consistency_weight,
                overall_weight=overall_weight,
            )
            meta_loss += loss

        meta_loss = meta_loss / meta_batch_size
        meta_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if meta_loss.item() < best_meta_loss:
            best_meta_loss = meta_loss.item()
            torch.save(model.state_dict(), save_path)

        if step % 10 == 0 or step == 1:
            print(f"[meta] step={step} meta_loss={meta_loss.item():.4f} best={best_meta_loss:.4f}")

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))


@torch.no_grad()
def predict_dataframe(
    model: T5EncoderAESModel,
    df: pd.DataFrame,
    tokenizer,
    normalizer: ScoreNormalizer,
    prompt_col: str,
    essay_col: str,
    prompt_text_col: Optional[str],
    canonical_traits: List[str],
    device: torch.device,
    max_length: int,
    batch_size: int,
    prompt_map: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    ds = AESDataset(
        df=df,
        tokenizer=tokenizer,
        prompt_col=prompt_col,
        essay_col=essay_col,
        prompt_text_col=prompt_text_col,
        normalizer=normalizer,
        canonical_traits=canonical_traits,
        max_length=max_length,
        prompt_map=prompt_map,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    model.eval()
    trait_preds = []
    overall_preds = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch["input_ids"], batch["attention_mask"])
        trait_preds.append(out["trait_scores"].cpu().numpy())
        overall_preds.append(out["overall_score"].cpu().numpy())
    return np.concatenate(trait_preds, axis=0), np.concatenate(overall_preds, axis=0)


def split_target_df(df: pd.DataFrame, support_frac: float, dev_frac: float, seed: int):
    labels = stratify_labels(df, OVERALL_COL)
    idx = np.arange(len(df))
    support_idx, remain_idx = train_test_split(
        idx,
        train_size=support_frac,
        stratify=labels,
        random_state=seed,
    )
    remain_df = df.iloc[remain_idx].copy()
    if dev_frac <= 0 or len(remain_df) < 3:
        return df.iloc[support_idx].copy(), None, remain_df
    remain_labels = stratify_labels(remain_df, OVERALL_COL)
    relative_dev_frac = dev_frac / max(1e-8, (1.0 - support_frac))
    relative_dev_frac = min(max(relative_dev_frac, 0.05), 0.95)
    dev_idx, test_idx = train_test_split(
        np.arange(len(remain_df)),
        train_size=relative_dev_frac,
        stratify=remain_labels,
        random_state=seed,
    )
    return df.iloc[support_idx].copy(), remain_df.iloc[dev_idx].copy(), remain_df.iloc[test_idx].copy()


def adapt_on_target(
    base_state_dict: Dict[str, torch.Tensor],
    cfg,
    target_df: pd.DataFrame,
    support_df: pd.DataFrame,
    dev_df: Optional[pd.DataFrame],
    query_df: pd.DataFrame,
    target_schema: TraitSchema,
    canonical_traits: List[str],
    output_dir: str,
    target_prompt_map: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = T5EncoderAESModel(model_name=cfg.model_name, num_traits=len(canonical_traits), dropout=cfg.dropout)
    model.load_state_dict(base_state_dict)

    if cfg.target_use_lora:
        model = apply_lora_to_encoder(model, r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout)

    freeze_bottom_encoder_layers(model, n_trainable_blocks=cfg.n_trainable_blocks)
    model.to(device)

    target_prompt_id = str(cfg.target_prompt_id)
    target_normalizer = ScoreNormalizer(canonical_traits=canonical_traits)
    target_normalizer.register_single_prompt(target_prompt_id, target_df)

    support_ds = AESDataset(
        support_df,
        tokenizer,
        cfg.prompt_col,
        cfg.essay_col,
        cfg.prompt_text_col,
        target_normalizer,
        canonical_traits,
        cfg.max_length,
        prompt_map=target_prompt_map,
    )
    support_loader = DataLoader(
        support_ds,
        batch_size=min(cfg.adapt_batch_size, len(support_ds)),
        shuffle=True,
        collate_fn=collate_batch,
    )

    dev_loader = None
    if dev_df is not None and len(dev_df) > 0:
        dev_ds = AESDataset(
            dev_df,
            tokenizer,
            cfg.prompt_col,
            cfg.essay_col,
            cfg.prompt_text_col,
            target_normalizer,
            canonical_traits,
            cfg.max_length,
            prompt_map=target_prompt_map,
        )
        dev_loader = DataLoader(
            dev_ds,
            batch_size=min(cfg.eval_batch_size, len(dev_ds)),
            shuffle=False,
            collate_fn=collate_batch,
        )

    optimizer = build_optimizer(model, lr=cfg.adapt_lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, cfg.adapt_epochs * len(support_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    best_loss = float("inf")
    best_path = os.path.join(output_dir, "best_target_model.pt")

    for epoch in range(1, cfg.adapt_epochs + 1):
        model.train()
        running = 0.0
        for batch in support_loader:
            batch = move_batch_to_device(batch, device)
            out = model(batch["input_ids"], batch["attention_mask"])
            loss = compute_joint_loss(
                out["trait_scores"],
                out["overall_score"],
                batch["traits"],
                batch["trait_mask"],
                batch["overall"],
                batch["overall_mask"],
                consistency_weight=cfg.consistency_weight,
                overall_weight=cfg.overall_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
        avg_train = running / max(1, len(support_loader))
        dev_loss = avg_train if dev_loader is None else evaluate_loss(
            model,
            dev_loader,
            device,
            cfg.consistency_weight,
            cfg.overall_weight,
        )
        print(f"[target-adapt] epoch={epoch} train_loss={avg_train:.4f} dev_loss={dev_loss:.4f}")
        if dev_loss < best_loss:
            best_loss = dev_loss
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    pred_traits, pred_overall = predict_dataframe(
        model,
        query_df,
        tokenizer,
        target_normalizer,
        cfg.prompt_col,
        cfg.essay_col,
        cfg.prompt_text_col,
        canonical_traits,
        device,
        cfg.max_length,
        cfg.eval_batch_size,
        prompt_map=target_prompt_map,
    )
    metrics = compute_metrics(
        query_df,
        pred_traits,
        pred_overall,
        target_normalizer,
        prompt_col=cfg.prompt_col,
        schema=target_schema,
        canonical_traits=canonical_traits,
    )

    with open(os.path.join(output_dir, "target_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# -----------------------------
# Entrypoint
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Trait-mapped AES meta-training (ANIL-style) + target LoRA adaptation."
    )
    p.add_argument("--source_csv", type=str, required=True)
    p.add_argument("--target_csv", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model_name", type=str, default="google/flan-t5-base")
    p.add_argument("--prompt_col", type=str, default="prompt_id")
    p.add_argument("--essay_col", type=str, default="essay")
    p.add_argument("--prompt_text_col", type=str, default=None)
    p.add_argument("--source_prompt_map", type=str, default=None, help="JSON string or path. Format: {\"essay_set_id\": \"prompt description\"}")
    p.add_argument("--target_prompt_map", type=str, default=None, help="JSON string or path. Format: {\"essay_set_id\": \"prompt description\"}")
    p.add_argument("--overall_col", type=str, default="overall")
    p.add_argument("--canonical_traits", nargs="+", default=CANONICAL_TRAITS)
    p.add_argument(
        "--source_trait_map",
        type=str,
        default=None,
        help="JSON string or path. Format: {\"source_column\": \"canonical_trait\"}",
    )
    p.add_argument(
        "--target_trait_map",
        type=str,
        default=None,
        help="JSON string or path. Format: {\"target_column\": \"canonical_trait\"}",
    )
    p.add_argument("--held_out_prompt_id", type=str, default=None)
    p.add_argument("--target_prompt_id", type=str, default="target")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--warmup_lr", type=float, default=2e-4)
    p.add_argument("--warmup_batch_size", type=int, default=8)

    p.add_argument("--meta_steps", type=int, default=200)
    p.add_argument("--meta_batch_size", type=int, default=2)
    p.add_argument("--support_size", type=int, default=8)
    p.add_argument("--query_size", type=int, default=16)
    p.add_argument("--inner_lr", type=float, default=1e-2)
    p.add_argument("--inner_steps", type=int, default=1)
    p.add_argument("--outer_lr", type=float, default=5e-5)

    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--consistency_weight", type=float, default=0.25)
    p.add_argument("--overall_weight", type=float, default=1.0)
    p.add_argument("--n_trainable_blocks", type=int, default=2)

    p.add_argument("--support_frac", type=float, default=0.6)
    p.add_argument("--dev_frac", type=float, default=0.2)
    p.add_argument("--adapt_epochs", type=int, default=20)
    p.add_argument("--adapt_lr", type=float, default=5e-5)
    p.add_argument("--adapt_batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--target_use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    print("Entering main function", flush=True)

    source_prompt_map = normalize_prompt_map_keys(safe_json_load(cfg.source_prompt_map)) or DEFAULT_SOURCE_PROMPT_MAP.copy()
    target_prompt_map = normalize_prompt_map_keys(safe_json_load(cfg.target_prompt_map)) or DEFAULT_TARGET_PROMPT_MAP.copy()

    source_df_raw = pd.read_csv(cfg.source_csv, sep="\t")
    source_schema = build_trait_schema(source_df_raw, cfg.source_trait_map, cfg.canonical_traits)
    source_df = materialize_canonical_columns(
        source_df_raw,
        schema=source_schema,
        prompt_col=cfg.prompt_col,
        overall_col=cfg.overall_col,
    )

    if cfg.target_csv is None and cfg.held_out_prompt_id is None:
        raise ValueError("Provide either --target_csv or --held_out_prompt_id.")

    if cfg.target_csv is None:
        train_df = source_df[source_df[cfg.prompt_col].astype(str) != str(cfg.held_out_prompt_id)].copy()
        target_df = source_df[source_df[cfg.prompt_col].astype(str) == str(cfg.held_out_prompt_id)].copy()
        target_schema = source_schema
        cfg.target_prompt_id = str(cfg.held_out_prompt_id)
        if not target_prompt_map:
            target_prompt_map = source_prompt_map
    else:
        train_df = source_df.copy()
        target_df_raw = pd.read_csv(cfg.target_csv, sep="\t")
        target_df_raw[cfg.prompt_col] = str(cfg.target_prompt_id)
        target_schema = build_trait_schema(target_df_raw, cfg.target_trait_map, cfg.canonical_traits)
        target_df = materialize_canonical_columns(
            target_df_raw,
            schema=target_schema,
            prompt_col=cfg.prompt_col,
            overall_col=cfg.overall_col,
        )

    with open(os.path.join(cfg.output_dir, "source_trait_schema.json"), "w", encoding="utf-8") as f:
        json.dump(source_schema.dataset_to_canonical, f, indent=2)
    with open(os.path.join(cfg.output_dir, "target_trait_schema.json"), "w", encoding="utf-8") as f:
        json.dump(target_schema.dataset_to_canonical, f, indent=2)
    with open(os.path.join(cfg.output_dir, "source_prompt_map.json"), "w", encoding="utf-8") as f:
        json.dump(source_prompt_map, f, indent=2)
    with open(os.path.join(cfg.output_dir, "target_prompt_map.json"), "w", encoding="utf-8") as f:
        json.dump(target_prompt_map, f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    normalizer = ScoreNormalizer(canonical_traits=cfg.canonical_traits)
    normalizer.fit(train_df, prompt_col=cfg.prompt_col)
    normalizer.save(os.path.join(cfg.output_dir, "source_prompt_stats.json"))

    train_labels = stratify_labels(train_df, OVERALL_COL)
    tr_idx, dv_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=0.1,
        stratify=train_labels,
        random_state=cfg.seed,
    )
    warm_train_df = train_df.iloc[tr_idx].copy()
    warm_dev_df = train_df.iloc[dv_idx].copy()

    warm_train_ds = AESDataset(
        warm_train_df,
        tokenizer,
        cfg.prompt_col,
        cfg.essay_col,
        cfg.prompt_text_col,
        normalizer,
        cfg.canonical_traits,
        cfg.max_length,
        prompt_map=source_prompt_map,
    )
    warm_dev_ds = AESDataset(
        warm_dev_df,
        tokenizer,
        cfg.prompt_col,
        cfg.essay_col,
        cfg.prompt_text_col,
        normalizer,
        cfg.canonical_traits,
        cfg.max_length,
        prompt_map=source_prompt_map,
    )
    warm_train_loader = DataLoader(
        warm_train_ds,
        batch_size=cfg.warmup_batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    warm_dev_loader = DataLoader(
        warm_dev_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    model = T5EncoderAESModel(
        model_name=cfg.model_name,
        num_traits=len(cfg.canonical_traits),
        dropout=cfg.dropout,
    )
    freeze_bottom_encoder_layers(model, n_trainable_blocks=cfg.n_trainable_blocks)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warm_ckpt = os.path.join(cfg.output_dir, "best_warmup_model.pt")
    print("Doing warmup training", flush=True)
    supervised_train(
        model,
        warm_train_loader,
        warm_dev_loader,
        device,
        lr=cfg.warmup_lr,
        epochs=cfg.warmup_epochs,
        weight_decay=cfg.weight_decay,
        consistency_weight=cfg.consistency_weight,
        overall_weight=cfg.overall_weight,
        save_path=warm_ckpt,
    )
    print("Warmup training completed", flush=True)
    print("Starting meta-training", flush=True)
    meta_train_anil(
        model,
        train_df=train_df,
        tokenizer=tokenizer,
        normalizer=normalizer,
        prompt_col=cfg.prompt_col,
        essay_col=cfg.essay_col,
        prompt_text_col=cfg.prompt_text_col,
        canonical_traits=cfg.canonical_traits,
        device=device,
        output_dir=cfg.output_dir,
        outer_lr=cfg.outer_lr,
        weight_decay=cfg.weight_decay,
        meta_steps=cfg.meta_steps,
        meta_batch_size=cfg.meta_batch_size,
        support_size=cfg.support_size,
        query_size=cfg.query_size,
        inner_lr=cfg.inner_lr,
        inner_steps=cfg.inner_steps,
        consistency_weight=cfg.consistency_weight,
        overall_weight=cfg.overall_weight,
        max_length=cfg.max_length,
        prompt_map=source_prompt_map,
    )

    base_state_dict = copy.deepcopy(model.state_dict())
    torch.save(base_state_dict, os.path.join(cfg.output_dir, "final_meta_state.pt"))
    print("Final meta state saved", flush=True)
    support_df, dev_df, query_df = split_target_df(
        target_df,
        support_frac=cfg.support_frac,
        dev_frac=cfg.dev_frac,
        seed=cfg.seed,
    )
    print("Starting target adaptation", flush=True)
    metrics = adapt_on_target(
        base_state_dict=base_state_dict,
        cfg=cfg,
        target_df=target_df,
        support_df=support_df,
        dev_df=dev_df,
        query_df=query_df,
        target_schema=target_schema,
        canonical_traits=cfg.canonical_traits,
        output_dir=cfg.output_dir,
        target_prompt_map=target_prompt_map,
    )

    print("\nFinal target metrics", flush=True)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
