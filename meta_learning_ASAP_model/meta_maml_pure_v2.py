from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
from collections import OrderedDict
from dataclasses import dataclass, asdict
from functools import lru_cache
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup

try:
    from torch.func import functional_call as torch_functional_call

    def functional_call(module, params, kwargs):
        return torch_functional_call(module, params, (), kwargs)
except Exception:
    from torch.nn.utils.stateless import functional_call as stateless_functional_call

    def functional_call(module, params, kwargs):
        return stateless_functional_call(module, params, (), kwargs)

print("Imported libraries successfully.", flush=True)

CANONICAL_TRAITS = [
    "content",
    "organization",
    "word_choice",
    "sentence_fluency",
    "conventions",
]
CANON_PREFIX = "__canon__"
MASK_PREFIX = "__mask__"
TRAIT_DESCRIPTIONS = {
    "content": "quality and development of ideas",
    "organization": "logical structure and progression",
    "word_choice": "precision and appropriateness of vocabulary",
    "sentence_fluency": "flow and readability of sentences",
    "conventions": "grammar, spelling, and punctuation",
}
DEFAULT_SOURCE_PROMPT_MAP: Dict[str, str] = {}
DEFAULT_TARGET_PROMPT_MAP: Dict[str, str] = {}
_DIGIT_RE = re.compile(r"\d")

SOURCE_SCORE_RANGES: Dict[Tuple[str, str], Tuple[int, int]] = {
    ("1", "content"): (1, 6),
    ("1", "organization"): (1, 6),
    ("1", "word_choice"): (1, 6),
    ("1", "sentence_fluency"): (1, 6),
    ("1", "conventions"): (1, 6),

    ("2", "content"): (1, 6),
    ("2", "organization"): (1, 6),
    ("2", "word_choice"): (1, 6),
    ("2", "sentence_fluency"): (1, 6),
    ("2", "conventions"): (1, 6),

    ("3", "content"): (0, 3),
    ("3", "prompt_adherence"): (0, 3),
    ("3", "language"): (0, 3),
    ("3", "narrativity"): (0, 3),

    ("4", "content"): (0, 3),
    ("4", "prompt_adherence"): (0, 3),
    ("4", "language"): (0, 3),
    ("4", "narrativity"): (0, 3),

    ("5", "content"): (0, 4),
    ("5", "prompt_adherence"): (0, 8),
    ("5", "language"): (0, 8),
    ("5", "narrativity"): (0, 8),

    ("6", "content"): (0, 8),
    ("6", "prompt_adherence"): (0, 4),
    ("6", "language"): (0, 4),
    ("6", "narrativity"): (0, 4),

    ("7", "content"): (0, 6),
    ("7", "organization"): (0, 6),
    ("7", "style"): (0, 6),
    ("7", "conventions"): (0, 6),

    ("8", "content"): (2, 12),
    ("8", "organization"): (2, 12),
    ("8", "voice"): (2, 12),
    ("8", "word_choice"): (2, 12),
    ("8", "sentence_fluency"): (2, 12),
    ("8", "conventions"): (2, 12),
}

TARGET_SCORE_RANGE_DEFAULT: Tuple[int, int] = (1, 6)

import gc

gc.collect()
torch.cuda.empty_cache()

def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def print_cuda_mem(tag: str) -> None:
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"{tag}: free={free / 1024**3:.2f} GB total={total / 1024**3:.2f} GB", flush=True)

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


@dataclass
class TraitRangeStats:
    mins: Dict[str, int]
    maxs: Dict[str, int]


class ScoreRangeStore:
    def __init__(self, canonical_traits: List[str]):
        self.canonical_traits = canonical_traits
        self.prompt_stats: Dict[str, TraitRangeStats] = {}
        self.global_mins: Dict[str, int] = {}
        self.global_maxs: Dict[str, int] = {}

    def fit_from_data(self, df: pd.DataFrame, prompt_col: str) -> None:
        self.prompt_stats = {}
        self._compute_global_stats_from_data(df)
        for prompt_id, group in df.groupby(prompt_col):
            self.prompt_stats[str(prompt_id)] = self._compute_stats_from_data(group)

    def fit_from_hardcoded_schema(
        self,
        df: pd.DataFrame,
        prompt_col: str,
        hardcoded_ranges: Dict[Tuple[str, str], Tuple[int, int]],
        default_range: Tuple[int, int] = (1, 6),
        fallback_to_data_for_missing: bool = True,
    ) -> None:
        self.prompt_stats = {}
        self._compute_global_stats_from_data(df)
        prompt_ids = [str(x) for x in sorted(df[prompt_col].astype(str).unique().tolist())]

        for prompt_id in prompt_ids:
            mins: Dict[str, int] = {}
            maxs: Dict[str, int] = {}
            prompt_df = df[df[prompt_col].astype(str) == str(prompt_id)]
            for canonical in self.canonical_traits:
                lookup_key = (str(prompt_id), canonical)
                if lookup_key in hardcoded_ranges:
                    lo, hi = hardcoded_ranges[lookup_key]
                    mins[canonical] = int(lo)
                    maxs[canonical] = int(hi)
                else:
                    if fallback_to_data_for_missing:
                        ccol = canon_trait_col(canonical)
                        mcol = mask_trait_col(canonical)
                        vals = pd.to_numeric(prompt_df.loc[prompt_df[mcol] > 0, ccol], errors="coerce").dropna()
                        if len(vals) > 0:
                            mins[canonical] = int(round(float(vals.min())))
                            maxs[canonical] = int(round(float(vals.max())))
                        else:
                            mins[canonical] = int(default_range[0])
                            maxs[canonical] = int(default_range[1])
                    else:
                        mins[canonical] = int(default_range[0])
                        maxs[canonical] = int(default_range[1])
                if maxs[canonical] < mins[canonical]:
                    maxs[canonical] = mins[canonical]
            self.prompt_stats[str(prompt_id)] = TraitRangeStats(mins=mins, maxs=maxs)

        for canonical in self.canonical_traits:
            lows = [stats.mins[canonical] for stats in self.prompt_stats.values()]
            highs = [stats.maxs[canonical] for stats in self.prompt_stats.values()]
            self.global_mins[canonical] = int(min(lows)) if lows else int(default_range[0])
            self.global_maxs[canonical] = int(max(highs)) if highs else int(default_range[1])

    def fit_target_uniform(self, df: pd.DataFrame, prompt_col: str, low: int = 1, high: int = 6) -> None:
        self.prompt_stats = {}
        prompt_ids = [str(x) for x in sorted(df[prompt_col].astype(str).unique().tolist())]
        for prompt_id in prompt_ids:
            self.prompt_stats[str(prompt_id)] = TraitRangeStats(
                mins={t: int(low) for t in self.canonical_traits},
                maxs={t: int(high) for t in self.canonical_traits},
            )
        self.global_mins = {t: int(low) for t in self.canonical_traits}
        self.global_maxs = {t: int(high) for t in self.canonical_traits}

    def _compute_global_stats_from_data(self, df: pd.DataFrame) -> None:
        for canonical in self.canonical_traits:
            ccol = canon_trait_col(canonical)
            mcol = mask_trait_col(canonical)
            vals = pd.to_numeric(df.loc[df[mcol] > 0, ccol], errors="coerce").dropna()
            if len(vals) == 0:
                self.global_mins[canonical] = 0
                self.global_maxs[canonical] = 1
            else:
                self.global_mins[canonical] = int(round(float(vals.min())))
                self.global_maxs[canonical] = int(round(float(vals.max())))
                if self.global_maxs[canonical] < self.global_mins[canonical]:
                    self.global_maxs[canonical] = self.global_mins[canonical]

    def _compute_stats_from_data(self, df: pd.DataFrame) -> TraitRangeStats:
        mins: Dict[str, int] = {}
        maxs: Dict[str, int] = {}
        for canonical in self.canonical_traits:
            ccol = canon_trait_col(canonical)
            mcol = mask_trait_col(canonical)
            vals = pd.to_numeric(df.loc[df[mcol] > 0, ccol], errors="coerce").dropna()
            if len(vals) == 0:
                mins[canonical] = self.global_mins.get(canonical, 0)
                maxs[canonical] = self.global_maxs.get(canonical, 1)
            else:
                mins[canonical] = int(round(float(vals.min())))
                maxs[canonical] = int(round(float(vals.max())))
                if maxs[canonical] < mins[canonical]:
                    maxs[canonical] = mins[canonical]
        return TraitRangeStats(mins=mins, maxs=maxs)

    def get_range(self, prompt_id: str, canonical_trait: str) -> Tuple[int, int]:
        stats = self.prompt_stats.get(str(prompt_id))
        if stats is None:
            return self.global_mins[canonical_trait], self.global_maxs[canonical_trait]
        return stats.mins[canonical_trait], stats.maxs[canonical_trait]

    def clip(self, prompt_id: str, canonical_trait: str, value: int) -> int:
        lo, hi = self.get_range(prompt_id, canonical_trait)
        return int(max(lo, min(hi, int(round(value)))))

    def midpoint(self, prompt_id: str, canonical_trait: str) -> int:
        lo, hi = self.get_range(prompt_id, canonical_trait)
        return int(round((lo + hi) / 2.0))

    def save(self, path: str) -> None:
        payload = {
            "prompt_stats": {k: asdict(v) for k, v in self.prompt_stats.items()},
            "global_mins": self.global_mins,
            "global_maxs": self.global_maxs,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def quadratic_weighted_kappa(y_true: List[int], y_pred: List[int]) -> float:
    if not y_true:
        return 0.0
    if len(set(y_true)) < 2 and len(set(y_pred)) < 2:
        return 1.0
    try:
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except Exception:
        return 0.0


def json_dump(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def count_trainable_parameters(model: torch.nn.Module) -> Dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "trainable_percent": float(100.0 * trainable / max(1, total)),
    }


def summarize_dataset(df: pd.DataFrame, prompt_col: str, canonical_traits: List[str], name: str) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "name": name,
        "rows": int(len(df)),
        "num_prompts": int(df[prompt_col].nunique()) if prompt_col in df.columns else 0,
        "rows_per_prompt": {str(k): int(v) for k, v in df[prompt_col].value_counts().sort_index().items()} if prompt_col in df.columns else {},
        "traits": {},
    }
    for canonical in canonical_traits:
        ccol = canon_trait_col(canonical)
        mcol = mask_trait_col(canonical)
        vals = pd.to_numeric(df.loc[df[mcol] > 0, ccol], errors="coerce").dropna()
        summary["traits"][canonical] = {
            "available_rows": int(len(vals)),
            "unique_scores": sorted(int(x) for x in vals.unique().tolist()) if len(vals) > 0 else [],
            "min": int(vals.min()) if len(vals) > 0 else None,
            "max": int(vals.max()) if len(vals) > 0 else None,
        }
    return summary


def build_monitor_df(df: pd.DataFrame, prompt_col: str, max_rows_per_prompt: int, seed: int) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for i, (_, group) in enumerate(df.groupby(prompt_col)):
        take = min(len(group), max_rows_per_prompt)
        parts.append(group.sample(n=take, random_state=seed + i).reset_index(drop=True))
    if not parts:
        return df.iloc[:0].copy()
    return pd.concat(parts, axis=0, ignore_index=True)


def summarize_prediction_df(pred_df: pd.DataFrame, canonical_traits: List[str], schema_label_map: Dict[str, str]) -> Dict[str, object]:
    out: Dict[str, object] = {"rows": int(len(pred_df)), "traits": {}}
    for canonical in canonical_traits:
        label = schema_label_map.get(canonical, canonical)
        pred_col = f"pred_{label}"
        gold_col = f"gold_{label}"
        if pred_col not in pred_df.columns:
            continue
        pred_vals = pd.to_numeric(pred_df[pred_col], errors="coerce").dropna()
        gold_vals = pd.to_numeric(pred_df[gold_col], errors="coerce").dropna() if gold_col in pred_df.columns else pd.Series([], dtype=float)
        out["traits"][label] = {
            "pred_unique_count": int(pred_vals.nunique()) if len(pred_vals) else 0,
            "pred_counts": {str(int(k)): int(v) for k, v in pred_vals.value_counts().sort_index().items()} if len(pred_vals) else {},
            "gold_counts": {str(int(k)): int(v) for k, v in gold_vals.value_counts().sort_index().items()} if len(gold_vals) else {},
            "pred_mean": float(pred_vals.mean()) if len(pred_vals) else None,
            "pred_std": float(pred_vals.std(ddof=0)) if len(pred_vals) else None,
        }
    if "generated_text" in pred_df.columns:
        out["sample_generations"] = pred_df["generated_text"].head(10).tolist()
    return out


def print_prediction_debug(prefix: str, pred_df: pd.DataFrame, canonical_traits: List[str], schema_label_map: Dict[str, str], max_examples: int) -> None:
    summary = summarize_prediction_df(pred_df, canonical_traits, schema_label_map)
    print(f"{prefix} prediction diversity:")
    for label, stats in summary.get("traits", {}).items():
        print(
            f"  {label}: unique={stats['pred_unique_count']} mean={stats['pred_mean']} std={stats['pred_std']} counts={stats['pred_counts']}"
        )
    if len(pred_df) > 0 and max_examples > 0:
        print(f"{prefix} sample predictions:")
        sample_cols = [c for c in pred_df.columns if c == 'generated_text' or c.startswith('gold_') or c.startswith('pred_')]
        for _, row in pred_df.head(max_examples).iterrows():
            payload = {c: row[c] for c in sample_cols}
            print("  ", payload)


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
            "No usable trait mapping was found. Provide a trait map or make your columns match the canonical names."
        )
    return TraitSchema(dataset_to_canonical=mapping, canonical_traits=canonical_traits)


def materialize_canonical_columns(df: pd.DataFrame, schema: TraitSchema, prompt_col: str) -> pd.DataFrame:
    out = df.copy()
    if prompt_col not in out.columns:
        raise ValueError(f"Missing prompt column '{prompt_col}'.")

    for canonical in schema.canonical_traits:
        out[canon_trait_col(canonical)] = np.nan
        out[mask_trait_col(canonical)] = 0.0

    for dataset_col, canonical in schema.dataset_to_canonical.items():
        if dataset_col not in out.columns:
            raise ValueError(f"Trait map expects dataset column '{dataset_col}', but it was not found.")
        out[canon_trait_col(canonical)] = pd.to_numeric(out[dataset_col], errors="coerce")
        out[mask_trait_col(canonical)] = out[canon_trait_col(canonical)].notna().astype(float)

    return out


def count_available_traits(df: pd.DataFrame, canonical_traits: List[str]) -> pd.Series:
    total = pd.Series(np.zeros(len(df)), index=df.index)
    for canonical in canonical_traits:
        total = total + pd.to_numeric(df[mask_trait_col(canonical)], errors="coerce").fillna(0)
    return total.astype(int)


def compute_avg_overall_from_traits(row: pd.Series, canonical_traits: List[str]) -> Optional[int]:
    vals: List[float] = []
    for canonical in canonical_traits:
        if float(row.get(mask_trait_col(canonical), 0.0)) > 0:
            vals.append(float(row[canon_trait_col(canonical)]))
    if not vals:
        return None
    return int(round(float(np.mean(vals))))


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


def trait_list_for_row(row: pd.Series, canonical_traits: List[str]) -> List[str]:
    return [t for t in canonical_traits if float(row.get(mask_trait_col(t), 0.0)) > 0]


def build_input_text(
    row: pd.Series,
    essay_col: str,
    prompt_col: str,
    prompt_text_col: Optional[str],
    prompt_map: Optional[Dict[str, str]],
    range_store: ScoreRangeStore,
    canonical_traits: List[str],
) -> str:
    prompt_id = str(row[prompt_col])
    prompt_text = resolve_prompt_text(row, prompt_col, prompt_text_col, prompt_map)
    traits = trait_list_for_row(row, canonical_traits)
    if not traits:
        traits = canonical_traits

    range_string = " ; ".join([
        f"{t} {range_store.get_range(prompt_id, t)[0]}-{range_store.get_range(prompt_id, t)[1]}" for t in traits
    ])
    desc_string = " ; ".join([f"{t} = {TRAIT_DESCRIPTIONS[t]}" for t in traits])

    lines = [
        "Score the essay traits.",
        "Return ONLY integers separated by | with no spaces.",
        f"Output order: {' | '.join(traits)}",
        f"Valid score ranges for this prompt: {range_string}",
        f"Trait meanings: {desc_string}",
    ]
    if prompt_text:
        lines.append(f"Prompt: {prompt_text}")
    else:
        lines.append(f"Prompt ID: {prompt_id}")
    lines.append("Essay:")
    lines.append(str(row[essay_col]))
    return "\n".join(lines)


def format_target_text_compact(
    row: pd.Series,
    canonical_traits: List[str],
    range_store: ScoreRangeStore,
    prompt_col: str,
) -> str:
    prompt_id = str(row[prompt_col])
    values: List[str] = []
    for canonical in canonical_traits:
        if float(row.get(mask_trait_col(canonical), 0.0)) <= 0:
            continue
        val = int(round(float(row[canon_trait_col(canonical)])))
        val = range_store.clip(prompt_id, canonical, val)
        values.append(str(val))
    if not values:
        raise ValueError("Cannot build target text for a row with zero available traits.")
    return "|".join(values)


def parse_generated_scores(text: str, expected_traits: List[str]) -> Dict[str, Optional[int]]:
    cleaned = text.strip().replace(" ", "").replace("\n", "")
    parts = [p for p in cleaned.split("|") if p != ""]
    parsed: Dict[str, Optional[int]] = {t: None for t in expected_traits}
    if len(parts) != len(expected_traits):
        digits = re.findall(r"-?\d+", cleaned)
        parts = digits[: len(expected_traits)]
    for idx, trait in enumerate(expected_traits):
        if idx >= len(parts):
            parsed[trait] = None
            continue
        try:
            parsed[trait] = int(parts[idx])
        except Exception:
            parsed[trait] = None
    return parsed


class Seq2SeqAESDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        prompt_col: str,
        essay_col: str,
        prompt_text_col: Optional[str],
        range_store: ScoreRangeStore,
        canonical_traits: List[str],
        max_input_length: int,
        max_target_length: int,
        prompt_map: Optional[Dict[str, str]] = None,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.prompt_col = prompt_col
        self.essay_col = essay_col
        self.prompt_text_col = prompt_text_col
        self.range_store = range_store
        self.canonical_traits = canonical_traits
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.prompt_map = normalize_prompt_map_keys(prompt_map)

    def __len__(self) -> int:
        return len(self.df)

    def _tokenize_target(self, text: str):
        try:
            encoded = self.tokenizer(
                text_target=text,
                padding=False,
                truncation=True,
                max_length=self.max_target_length,
            )
            return encoded["input_ids"]
        except TypeError:
            with self.tokenizer.as_target_tokenizer():
                encoded = self.tokenizer(
                    text,
                    padding=False,
                    truncation=True,
                    max_length=self.max_target_length,
                )
            return encoded["input_ids"]

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        input_text = build_input_text(
            row=row,
            essay_col=self.essay_col,
            prompt_col=self.prompt_col,
            prompt_text_col=self.prompt_text_col,
            prompt_map=self.prompt_map,
            range_store=self.range_store,
            canonical_traits=self.canonical_traits,
        )
        target_text = format_target_text_compact(
            row=row,
            canonical_traits=self.canonical_traits,
            range_store=self.range_store,
            prompt_col=self.prompt_col,
        )
        model_inputs = self.tokenizer(
            input_text,
            padding=False,
            truncation=True,
            max_length=self.max_input_length,
        )
        labels = self._tokenize_target(target_text)
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels,
            "prompt_id": str(row[self.prompt_col]),
            "target_text": target_text,
        }


def collate_seq2seq_batch(features: List[Dict], tokenizer) -> Dict:
    input_batch = tokenizer.pad(
        [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features],
        padding=True,
        return_tensors="pt",
    )
    label_batch = tokenizer.pad(
        [{"input_ids": f["labels"]} for f in features],
        padding=True,
        return_tensors="pt",
    )
    labels = label_batch["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    return {
        "input_ids": input_batch["input_ids"],
        "attention_mask": input_batch["attention_mask"],
        "labels": labels,
        "prompt_ids": [f["prompt_id"] for f in features],
        "target_texts": [f["target_text"] for f in features],
    }


class EpisodeSampler:
    def __init__(self, df: pd.DataFrame, prompt_col: str, seed: int = 42, mode: str = "balanced"):
        self.df = df.reset_index(drop=True).copy()
        self.prompt_col = prompt_col
        self.rng = np.random.default_rng(seed)
        self.groups = {str(k): g.copy().reset_index(drop=True) for k, g in self.df.groupby(prompt_col)}
        self.mode = mode
        self._prompt_cycle: List[str] = []

    def valid_prompts(self, support_size: int, query_size: int) -> List[str]:
        needed = support_size + query_size
        return [pid for pid, g in self.groups.items() if len(g) >= needed]

    def _next_balanced_prompt(self, support_size: int, query_size: int) -> str:
        prompts = self.valid_prompts(support_size, query_size)
        if not prompts:
            raise ValueError("No prompt has enough examples for the requested support/query sizes.")
        if not self._prompt_cycle:
            self._prompt_cycle = prompts.copy()
            self.rng.shuffle(self._prompt_cycle)
        return self._prompt_cycle.pop(0)

    def sample(self, support_size: int, query_size: int) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
        prompts = self.valid_prompts(support_size, query_size)
        if not prompts:
            raise ValueError("No prompt has enough examples for the requested support/query sizes.")
        if self.mode == "balanced":
            prompt_id = self._next_balanced_prompt(support_size, query_size)
        else:
            prompt_id = str(self.rng.choice(prompts))
        group = self.groups[prompt_id]
        indices = self.rng.choice(len(group), size=support_size + query_size, replace=False)
        support_idx = indices[:support_size]
        query_idx = indices[support_size:]
        support_df = group.iloc[support_idx].reset_index(drop=True)
        query_df = group.iloc[query_idx].reset_index(drop=True)
        return prompt_id, support_df, query_df


def build_dataset(
    df: pd.DataFrame,
    tokenizer,
    prompt_col: str,
    essay_col: str,
    prompt_text_col: Optional[str],
    range_store: ScoreRangeStore,
    canonical_traits: List[str],
    max_input_length: int,
    max_target_length: int,
    prompt_map: Optional[Dict[str, str]] = None,
) -> Seq2SeqAESDataset:
    return Seq2SeqAESDataset(
        df=df,
        tokenizer=tokenizer,
        prompt_col=prompt_col,
        essay_col=essay_col,
        prompt_text_col=prompt_text_col,
        range_store=range_store,
        canonical_traits=canonical_traits,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        prompt_map=prompt_map,
    )


def make_batch(
    df: pd.DataFrame,
    tokenizer,
    prompt_col: str,
    essay_col: str,
    prompt_text_col: Optional[str],
    range_store: ScoreRangeStore,
    canonical_traits: List[str],
    max_input_length: int,
    max_target_length: int,
    prompt_map: Optional[Dict[str, str]],
) -> Dict:
    ds = build_dataset(
        df=df,
        tokenizer=tokenizer,
        prompt_col=prompt_col,
        essay_col=essay_col,
        prompt_text_col=prompt_text_col,
        range_store=range_store,
        canonical_traits=canonical_traits,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        prompt_map=prompt_map,
    )
    items = [ds[i] for i in range(len(ds))]
    return collate_seq2seq_batch(items, tokenizer)


def read_table_auto(path: str, sep: Optional[str]) -> pd.DataFrame:
    if sep is not None and sep != "auto":
        return pd.read_csv(path, sep=sep)
    return pd.read_csv(path, sep=None, engine="python", on_bad_lines="warn")


def safe_train_test_split_indices(
    n: int,
    test_size: float,
    labels: Optional[np.ndarray],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    if n <= 1 or test_size <= 0:
        return idx, np.array([], dtype=int)
    test_n = max(1, int(round(n * test_size)))
    if test_n >= n:
        test_n = n - 1
    if test_n <= 0:
        return idx, np.array([], dtype=int)
    stratify = None
    if labels is not None and len(np.unique(labels)) > 1:
        counts = {int(x): int((labels == x).sum()) for x in np.unique(labels)}
        if min(counts.values()) >= 2:
            stratify = labels
    try:
        tr, te = train_test_split(idx, test_size=test_n, random_state=seed, stratify=stratify)
        return np.array(tr), np.array(te)
    except Exception:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(idx)
        te = perm[:test_n]
        tr = perm[test_n:]
        return np.array(tr), np.array(te)


def overall_labels_for_split(df: pd.DataFrame, canonical_traits: List[str]) -> np.ndarray:
    vals: List[int] = []
    for _, row in df.iterrows():
        overall = compute_avg_overall_from_traits(row, canonical_traits)
        vals.append(0 if overall is None else int(overall))
    return np.array(vals)


def split_target_by_prompt(
    df: pd.DataFrame,
    prompt_col: str,
    support_frac: float,
    dev_frac: float,
    canonical_traits: List[str],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    query_frac = 1.0 - support_frac - dev_frac
    if query_frac <= 0:
        raise ValueError("support_frac + dev_frac must be < 1.")
    support_parts: List[pd.DataFrame] = []
    dev_parts: List[pd.DataFrame] = []
    query_parts: List[pd.DataFrame] = []

    for i, (_, group) in enumerate(df.groupby(prompt_col)):
        group = group.sample(frac=1.0, random_state=seed + i).reset_index(drop=True)
        n = len(group)
        labels = overall_labels_for_split(group, canonical_traits)
        support_idx, temp_idx = safe_train_test_split_indices(n, 1.0 - support_frac, labels, seed + i)
        support_group = group.iloc[support_idx].reset_index(drop=True)
        temp_group = group.iloc[temp_idx].reset_index(drop=True)
        if len(temp_group) == 0:
            dev_group = temp_group.copy()
            query_group = temp_group.copy()
        else:
            temp_labels = overall_labels_for_split(temp_group, canonical_traits)
            dev_share = dev_frac / (dev_frac + query_frac)
            dev_idx, query_idx = safe_train_test_split_indices(len(temp_group), 1.0 - dev_share, temp_labels, seed + 1000 + i)
            dev_group = temp_group.iloc[dev_idx].reset_index(drop=True)
            query_group = temp_group.iloc[query_idx].reset_index(drop=True)
        support_parts.append(support_group)
        dev_parts.append(dev_group)
        query_parts.append(query_group)

    support_df = pd.concat(support_parts, axis=0, ignore_index=True) if support_parts else df.iloc[:0].copy()
    dev_df = pd.concat(dev_parts, axis=0, ignore_index=True) if dev_parts else df.iloc[:0].copy()
    query_df = pd.concat(query_parts, axis=0, ignore_index=True) if query_parts else df.iloc[:0].copy()
    return support_df, dev_df, query_df


def configure_trainable_params(
    model: T5ForConditionalGeneration,
    n_trainable_encoder_blocks: int,
    n_trainable_decoder_blocks: int,
    train_final_layer_norms: bool,
    train_lm_head: bool,
) -> None:
    for p in model.parameters():
        p.requires_grad = False

    enc_blocks = model.encoder.block
    dec_blocks = model.decoder.block
    enc_cutoff = max(0, len(enc_blocks) - n_trainable_encoder_blocks)
    dec_cutoff = max(0, len(dec_blocks) - n_trainable_decoder_blocks)

    for idx, block in enumerate(enc_blocks):
        if idx >= enc_cutoff:
            for p in block.parameters():
                p.requires_grad = True
    for idx, block in enumerate(dec_blocks):
        if idx >= dec_cutoff:
            for p in block.parameters():
                p.requires_grad = True

    if train_final_layer_norms:
        if hasattr(model.encoder, "final_layer_norm"):
            for p in model.encoder.final_layer_norm.parameters():
                p.requires_grad = True
        if hasattr(model.decoder, "final_layer_norm"):
            for p in model.decoder.final_layer_norm.parameters():
                p.requires_grad = True
    if train_lm_head:
        for p in model.lm_head.parameters():
            p.requires_grad = True


def trainable_param_dict(model: T5ForConditionalGeneration) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, p) for name, p in model.named_parameters() if p.requires_grad)


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


@lru_cache(maxsize=50000)
def _token_has_digit(token_id: int, model_name: str, tokenizer) -> bool:
    token_text = tokenizer.decode([token_id], skip_special_tokens=False)
    return bool(_DIGIT_RE.search(token_text))


def weighted_seq2seq_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
    model_name: str,
    score_token_weight: float,
    separator_token_weight: float,
) -> torch.Tensor:
    vocab_size = logits.size(-1)
    flat_logits = logits.reshape(-1, vocab_size)
    flat_labels = labels.reshape(-1)
    per_token = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction="none").view_as(labels)

    weights = torch.zeros_like(labels, dtype=logits.dtype)
    valid_mask = labels.ne(-100)
    if valid_mask.any():
        valid_ids = labels[valid_mask].detach().cpu().tolist()
        valid_weights = [
            score_token_weight if _token_has_digit(int(tok_id), model_name, tokenizer) else separator_token_weight
            for tok_id in valid_ids
        ]
        weights[valid_mask] = torch.tensor(valid_weights, dtype=logits.dtype, device=labels.device)

    denom = weights[valid_mask].sum().clamp_min(1e-8)
    return (per_token * weights).sum() / denom


def model_forward_with_params(model: T5ForConditionalGeneration, params: OrderedDict[str, torch.Tensor], batch: Dict):
    return functional_call(
        model,
        params,
        {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        },
    )


class ConstraintFactory:
    def __init__(self, tokenizer, decoder_start_token_id: int, eos_token_id: int):
        self.tokenizer = tokenizer
        self.decoder_start_token_id = decoder_start_token_id
        self.eos_token_id = eos_token_id
        self.cache: Dict[Tuple[str, Tuple[str, ...]], Tuple[List[str], List[List[int]]]] = {}

    def candidate_texts(self, prompt_id: str, traits: List[str], range_store: ScoreRangeStore) -> List[str]:
        ranges = [range_store.get_range(prompt_id, t) for t in traits]
        texts: List[str] = []
        for combo in product(*[range(lo, hi + 1) for lo, hi in ranges]):
            texts.append("|".join(str(v) for v in combo))
        return texts

    def candidate_token_sequences(self, prompt_id: str, traits: List[str], range_store: ScoreRangeStore) -> Tuple[List[str], List[List[int]]]:
        key = (str(prompt_id), tuple(traits))
        if key not in self.cache:
            texts = self.candidate_texts(prompt_id, traits, range_store)
            seqs: List[List[int]] = []
            for text in texts:
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                seqs.append([self.decoder_start_token_id] + token_ids + [self.eos_token_id])
            self.cache[key] = (texts, seqs)
        return self.cache[key]

    def prefix_allowed_tokens_fn(self, prompt_id: str, traits: List[str], range_store: ScoreRangeStore):
        _, seqs = self.candidate_token_sequences(prompt_id, traits, range_store)

        def fn(_batch_id: int, sent: torch.Tensor) -> List[int]:
            prefix = sent.tolist()
            allowed = set()
            for seq in seqs:
                if len(prefix) <= len(seq) and seq[: len(prefix)] == prefix:
                    if len(prefix) < len(seq):
                        allowed.add(seq[len(prefix)])
            if not allowed:
                return [self.eos_token_id]
            return sorted(allowed)

        return fn


def evaluate_generation(
    model: T5ForConditionalGeneration,
    tokenizer,
    df: pd.DataFrame,
    prompt_col: str,
    essay_col: str,
    prompt_text_col: Optional[str],
    canonical_traits: List[str],
    schema_label_map: Optional[Dict[str, str]],
    device: torch.device,
    range_store: ScoreRangeStore,
    max_input_length: int,
    generation_max_new_tokens: int,
    prompt_map: Optional[Dict[str, str]],
    constrained_num_beams: int,
    constraint_factory: ConstraintFactory,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    if schema_label_map is None:
        schema_label_map = {t: t for t in canonical_traits}
    model.eval()
    rows: List[Dict] = []
    per_trait_true: Dict[str, List[int]] = {t: [] for t in canonical_traits}
    per_trait_pred: Dict[str, List[int]] = {t: [] for t in canonical_traits}
    overall_true: List[int] = []
    overall_pred: List[int] = []
    parse_success = 0
    parse_total = 0

    for _, row in df.reset_index(drop=True).iterrows():
        expected_traits = trait_list_for_row(row, canonical_traits)
        if not expected_traits:
            continue
        input_text = build_input_text(
            row=row,
            essay_col=essay_col,
            prompt_col=prompt_col,
            prompt_text_col=prompt_text_col,
            prompt_map=prompt_map,
            range_store=range_store,
            canonical_traits=canonical_traits,
        )
        encoded = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        prompt_id = str(row[prompt_col])
        candidate_texts, candidate_token_seqs = constraint_factory.candidate_token_sequences(prompt_id, expected_traits, range_store)
        max_candidate_new = max(len(seq) - 1 for seq in candidate_token_seqs)
        prefix_fn = constraint_factory.prefix_allowed_tokens_fn(prompt_id, expected_traits, range_store)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=min(generation_max_new_tokens, max_candidate_new),
                num_beams=min(max(1, constrained_num_beams), max(1, len(candidate_texts))),
                do_sample=False,
                prefix_allowed_tokens_fn=prefix_fn,
            )
        gen_text = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        parsed = parse_generated_scores(gen_text, expected_traits)
        row_out: Dict[str, object] = {
            "prompt_id": prompt_id,
            "generated_text": gen_text,
            "input_text": input_text,
            "gold_compact_text": format_target_text_compact(
                row=row,
                canonical_traits=canonical_traits,
                range_store=range_store,
                prompt_col=prompt_col,
            ),
        }

        gold_vals: List[int] = []
        pred_vals: List[int] = []
        all_found = True
        for canonical in expected_traits:
            gold = int(round(float(row[canon_trait_col(canonical)])))
            pred_raw = parsed.get(canonical)
            if pred_raw is None:
                all_found = False
                pred = range_store.midpoint(prompt_id, canonical)
            else:
                pred = range_store.clip(prompt_id, canonical, pred_raw)
            per_trait_true[canonical].append(gold)
            per_trait_pred[canonical].append(pred)
            gold_vals.append(gold)
            pred_vals.append(pred)
            row_out[f"gold_{schema_label_map.get(canonical, canonical)}"] = gold
            row_out[f"pred_{schema_label_map.get(canonical, canonical)}"] = pred

        parse_total += 1
        if all_found:
            parse_success += 1

        true_overall = int(round(float(np.mean(gold_vals))))
        pred_overall = int(round(float(np.mean(pred_vals))))
        overall_true.append(true_overall)
        overall_pred.append(pred_overall)
        row_out["gold_overall_from_traits"] = true_overall
        row_out["pred_overall_from_traits"] = pred_overall
        rows.append(row_out)

    metrics: Dict[str, float] = {}
    qwk_values: List[float] = []
    for canonical in canonical_traits:
        if per_trait_true[canonical]:
            qwk = quadratic_weighted_kappa(per_trait_true[canonical], per_trait_pred[canonical])
            metrics[f"qwk_{schema_label_map.get(canonical, canonical)}"] = qwk
            qwk_values.append(qwk)
    metrics["mean_trait_qwk"] = float(np.mean(qwk_values)) if qwk_values else 0.0
    metrics["overall_qwk_from_traits"] = quadratic_weighted_kappa(overall_true, overall_pred) if overall_true else 0.0
    metrics["parse_success_rate"] = (parse_success / parse_total) if parse_total else 0.0
    pred_df = pd.DataFrame(rows)
    return metrics, pred_df


def supervised_warmup(
    model: T5ForConditionalGeneration,
    train_loader: DataLoader,
    dev_df: Optional[pd.DataFrame],
    tokenizer,
    range_store: ScoreRangeStore,
    prompt_col: str,
    essay_col: str,
    prompt_text_col: Optional[str],
    canonical_traits: List[str],
    prompt_map: Optional[Dict[str, str]],
    schema_label_map: Dict[str, str],
    device: torch.device,
    lr: float,
    weight_decay: float,
    epochs: int,
    output_dir: str,
    max_input_length: int,
    generation_max_new_tokens: int,
    score_token_weight: float,
    separator_token_weight: float,
    constrained_num_beams: int,
    constraint_factory: ConstraintFactory,
    model_name: str,
    debug_mode: bool = False,
    debug_examples: int = 6,
) -> str:
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    steps = max(1, epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(0.1 * steps)), num_training_steps=steps)
    best_metric = -float("inf")
    best_path = os.path.join(output_dir, "best_warmup_seq2seq_compact.pt")
    print_cuda_mem("warmup before model.to")
    if next(model.parameters()).device != device:
        clear_cuda()
        print_cuda_mem("adapt before model.to")
    model.to(device)
    print_cuda_mem("adapt after model.to")
    print_cuda_mem("warmup after model.to")

    for epoch in range(epochs):
        model.train()
        train_losses: List[float] = []
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = weighted_seq2seq_loss(
                outputs.logits,
                batch["labels"],
                tokenizer=tokenizer,
                model_name=model_name,
                score_token_weight=score_token_weight,
                separator_token_weight=separator_token_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.item()))

        if dev_df is None or len(dev_df) == 0:
            torch.save(model.state_dict(), best_path)
            continue

        metrics, pred_df = evaluate_generation(
            model=model,
            tokenizer=tokenizer,
            df=dev_df,
            prompt_col=prompt_col,
            essay_col=essay_col,
            prompt_text_col=prompt_text_col,
            canonical_traits=canonical_traits,
            schema_label_map=schema_label_map,
            device=device,
            range_store=range_store,
            max_input_length=max_input_length,
            generation_max_new_tokens=generation_max_new_tokens,
            prompt_map=prompt_map,
            constrained_num_beams=constrained_num_beams,
            constraint_factory=constraint_factory,
        )
        mean_train = float(np.mean(train_losses)) if train_losses else float("inf")
        metric_value = metrics.get("mean_trait_qwk", 0.0)
        print(
            f"Warmup epoch {epoch + 1} | train_loss={mean_train:.4f} | "
            f"dev_mean_trait_qwk={metric_value:.4f} | dev_parse={metrics.get('parse_success_rate', 0.0):.4f}"
        )
        if debug_mode:
            pred_df.to_csv(os.path.join(output_dir, f"warmup_dev_predictions_epoch_{epoch + 1}.csv"), index=False)
            debug_summary = summarize_prediction_df(pred_df, canonical_traits, schema_label_map)
            json_dump(os.path.join(output_dir, f"warmup_dev_debug_epoch_{epoch + 1}.json"), debug_summary)
            print_prediction_debug(f"Warmup epoch {epoch + 1}", pred_df, canonical_traits, schema_label_map, debug_examples)
        if metric_value > best_metric:
            best_metric = metric_value
            torch.save(model.state_dict(), best_path)

    return best_path


def meta_train_maml(
    model: T5ForConditionalGeneration,
    train_df: pd.DataFrame,
    tokenizer,
    range_store: ScoreRangeStore,
    prompt_col: str,
    essay_col: str,
    prompt_text_col: Optional[str],
    canonical_traits: List[str],
    device: torch.device,
    output_dir: str,
    meta_steps: int,
    meta_batch_size: int,
    support_size: int,
    query_size: int,
    inner_lr: float,
    inner_steps: int,
    outer_lr: float,
    weight_decay: float,
    max_input_length: int,
    max_target_length: int,
    prompt_map: Optional[Dict[str, str]],
    score_token_weight: float,
    separator_token_weight: float,
    model_name: str,
    generation_max_new_tokens: int,
    constrained_num_beams: int,
    constraint_factory: ConstraintFactory,
    prompt_sampling: str = "balanced",
    meta_eval_every: int = 0,
    source_monitor_df: Optional[pd.DataFrame] = None,
    meta_select_metric: str = "loss",
    debug_mode: bool = False,
    debug_examples: int = 6,
) -> str:
    sampler = EpisodeSampler(train_df, prompt_col=prompt_col, seed=42, mode=prompt_sampling)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=outer_lr, weight_decay=weight_decay)
    if next(model.parameters()).device != device:
        clear_cuda()
        model.to(device)
    best_path = os.path.join(output_dir, "best_meta_maml_seq2seq_compact.pt")
    best_loss = float("inf")
    best_metric = -float("inf")

    for step in range(1, meta_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        task_losses: List[torch.Tensor] = []
        task_ids: List[str] = []
        for _ in range(meta_batch_size):
            prompt_id, support_df, query_df = sampler.sample(support_size=support_size, query_size=query_size)
            support_batch = move_batch_to_device(
                make_batch(
                    df=support_df,
                    tokenizer=tokenizer,
                    prompt_col=prompt_col,
                    essay_col=essay_col,
                    prompt_text_col=prompt_text_col,
                    range_store=range_store,
                    canonical_traits=canonical_traits,
                    max_input_length=max_input_length,
                    max_target_length=max_target_length,
                    prompt_map=prompt_map,
                ),
                device,
            )
            query_batch = move_batch_to_device(
                make_batch(
                    df=query_df,
                    tokenizer=tokenizer,
                    prompt_col=prompt_col,
                    essay_col=essay_col,
                    prompt_text_col=prompt_text_col,
                    range_store=range_store,
                    canonical_traits=canonical_traits,
                    max_input_length=max_input_length,
                    max_target_length=max_target_length,
                    prompt_map=prompt_map,
                ),
                device,
            )

            if debug_mode and step == 1 and len(task_ids) == 0:
                print_cuda_mem("before first support/query forward")
                print("support batch shape:", tuple(support_batch["input_ids"].shape), flush=True)
                print("query batch shape:", tuple(query_batch["input_ids"].shape), flush=True)

            fast_weights = trainable_param_dict(model)
            for _inner in range(inner_steps):
                support_outputs = model_forward_with_params(model, fast_weights, support_batch)
                support_loss = weighted_seq2seq_loss(
                    support_outputs.logits,
                    support_batch["labels"],
                    tokenizer=tokenizer,
                    model_name=model_name,
                    score_token_weight=score_token_weight,
                    separator_token_weight=separator_token_weight,
                )
                grads = torch.autograd.grad(support_loss, list(fast_weights.values()), create_graph=True)
                fast_weights = OrderedDict(
                    (name, param - inner_lr * grad)
                    for (name, param), grad in zip(fast_weights.items(), grads)
                )

            query_outputs = model_forward_with_params(model, fast_weights, query_batch)
            task_loss = weighted_seq2seq_loss(
                query_outputs.logits,
                query_batch["labels"],
                tokenizer=tokenizer,
                model_name=model_name,
                score_token_weight=score_token_weight,
                separator_token_weight=separator_token_weight,
            )
            task_losses.append(task_loss)
            task_ids.append(prompt_id)

        meta_loss = torch.stack(task_losses).mean()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        loss_value = float(meta_loss.item())
        if step % 5 == 0 or step == 1 or step == meta_steps:
            print(f"Meta step {step}/{meta_steps} | meta_loss={loss_value:.4f} | prompts={task_ids}")

        monitor_metric = None
        if source_monitor_df is not None and len(source_monitor_df) > 0 and meta_eval_every > 0 and (step % meta_eval_every == 0 or step == meta_steps):
            metrics, pred_df = evaluate_generation(
                model=model,
                tokenizer=tokenizer,
                df=source_monitor_df,
                prompt_col=prompt_col,
                essay_col=essay_col,
                prompt_text_col=prompt_text_col,
                canonical_traits=canonical_traits,
                schema_label_map={t: t for t in canonical_traits},
                device=device,
                range_store=range_store,
                max_input_length=max_input_length,
                generation_max_new_tokens=generation_max_new_tokens,
                prompt_map=prompt_map,
                constrained_num_beams=constrained_num_beams,
                constraint_factory=constraint_factory,
            )
            monitor_metric = metrics.get("mean_trait_qwk", 0.0)
            print(
                f"  Source monitor | mean_trait_qwk={monitor_metric:.4f} | overall_qwk={metrics.get('overall_qwk_from_traits', 0.0):.4f} | parse={metrics.get('parse_success_rate', 0.0):.4f}"
            )
            pred_df.to_csv(os.path.join(output_dir, f"meta_monitor_predictions_step_{step}.csv"), index=False)
            json_dump(os.path.join(output_dir, f"meta_monitor_metrics_step_{step}.json"), metrics)
            if debug_mode:
                json_dump(os.path.join(output_dir, f"meta_monitor_debug_step_{step}.json"), summarize_prediction_df(pred_df, canonical_traits, {t: t for t in canonical_traits}))
                print_prediction_debug(f"Meta step {step}", pred_df, canonical_traits, {t: t for t in canonical_traits}, debug_examples)

        should_save = False
        if meta_select_metric == "mean_trait_qwk" and monitor_metric is not None:
            if monitor_metric > best_metric:
                best_metric = monitor_metric
                should_save = True
        else:
            if loss_value < best_loss:
                best_loss = loss_value
                should_save = True
        if should_save:
            torch.save(model.state_dict(), best_path)

    return best_path


def adapt_on_target_generation(
    model: T5ForConditionalGeneration,
    tokenizer,
    support_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    query_df: pd.DataFrame,
    prompt_col: str,
    essay_col: str,
    prompt_text_col: Optional[str],
    canonical_traits: List[str],
    schema_label_map: Dict[str, str],
    device: torch.device,
    output_dir: str,
    adapt_lr: float,
    adapt_epochs: int,
    adapt_batch_size: int,
    max_input_length: int,
    max_target_length: int,
    generation_max_new_tokens: int,
    prompt_map: Optional[Dict[str, str]],
    train_final_layer_norms: bool,
    train_lm_head: bool,
    n_trainable_encoder_blocks: int,
    n_trainable_decoder_blocks: int,
    score_token_weight: float,
    separator_token_weight: float,
    constrained_num_beams: int,
    constraint_factory: ConstraintFactory,
    model_name: str,
    debug_mode: bool = False,
    debug_examples: int = 6,
) -> Dict[str, float]:
    norm_df = pd.concat([support_df, dev_df], axis=0, ignore_index=True) if len(dev_df) > 0 else support_df.copy()
    target_ranges = ScoreRangeStore(canonical_traits=canonical_traits)
    target_ranges.fit_target_uniform(norm_df, prompt_col=prompt_col, low=1, high=6)
    target_ranges.save(os.path.join(output_dir, "target_prompt_ranges.json"))

    support_ds = build_dataset(
        support_df, tokenizer, prompt_col, essay_col, prompt_text_col, target_ranges,
        canonical_traits, max_input_length, max_target_length, prompt_map
    )
    support_loader = DataLoader(
        support_ds,
        batch_size=adapt_batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_seq2seq_batch(x, tokenizer),
    )

    model = copy.deepcopy(model.cpu())
    clear_cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    configure_trainable_params(
        model,
        n_trainable_encoder_blocks=n_trainable_encoder_blocks,
        n_trainable_decoder_blocks=n_trainable_decoder_blocks,
        train_final_layer_norms=train_final_layer_norms,
        train_lm_head=train_lm_head,
    )
    model.to(device)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=adapt_lr)
    best_state = copy.deepcopy(model.state_dict())
    best_metric = -float("inf")

    for epoch in range(adapt_epochs):
        model.train()
        epoch_losses: List[float] = []
        for batch in support_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = weighted_seq2seq_loss(
                outputs.logits,
                batch["labels"],
                tokenizer=tokenizer,
                model_name=model_name,
                score_token_weight=score_token_weight,
                separator_token_weight=separator_token_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        metrics, pred_df = evaluate_generation(
            model=model,
            tokenizer=tokenizer,
            df=dev_df,
            prompt_col=prompt_col,
            essay_col=essay_col,
            prompt_text_col=prompt_text_col,
            canonical_traits=canonical_traits,
            schema_label_map=schema_label_map,
            device=device,
            range_store=target_ranges,
            max_input_length=max_input_length,
            generation_max_new_tokens=generation_max_new_tokens,
            prompt_map=prompt_map,
            constrained_num_beams=constrained_num_beams,
            constraint_factory=constraint_factory,
        )
        mean_train = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        metric_value = metrics.get("mean_trait_qwk", 0.0)
        print(
            f"Adapt epoch {epoch + 1}/{adapt_epochs} | train_loss={mean_train:.4f} | "
            f"dev_mean_trait_qwk={metric_value:.4f} | dev_overall_qwk={metrics.get('overall_qwk_from_traits', 0.0):.4f} | "
            f"dev_parse={metrics.get('parse_success_rate', 0.0):.4f}"
        )
        pred_df.to_csv(os.path.join(output_dir, f"adapt_dev_predictions_epoch_{epoch + 1}.csv"), index=False)
        if debug_mode:
            json_dump(os.path.join(output_dir, f"adapt_dev_debug_epoch_{epoch + 1}.json"), summarize_prediction_df(pred_df, canonical_traits, schema_label_map))
            print_prediction_debug(f"Adapt epoch {epoch + 1}", pred_df, canonical_traits, schema_label_map, debug_examples)
        if metric_value > best_metric:
            best_metric = metric_value
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(output_dir, "best_target_adapted_seq2seq_compact.pt"))

    metrics, pred_df = evaluate_generation(
        model=model,
        tokenizer=tokenizer,
        df=query_df,
        prompt_col=prompt_col,
        essay_col=essay_col,
        prompt_text_col=prompt_text_col,
        canonical_traits=canonical_traits,
        schema_label_map=schema_label_map,
        device=device,
        range_store=target_ranges,
        max_input_length=max_input_length,
        generation_max_new_tokens=generation_max_new_tokens,
        prompt_map=prompt_map,
        constrained_num_beams=constrained_num_beams,
        constraint_factory=constraint_factory,
    )
    pred_df.to_csv(os.path.join(output_dir, "target_query_predictions.csv"), index=False)
    json_dump(os.path.join(output_dir, "target_query_debug_summary.json"), summarize_prediction_df(pred_df, canonical_traits, schema_label_map))
    if debug_mode:
        print_prediction_debug("Final query", pred_df, canonical_traits, schema_label_map, debug_examples)
    return metrics


def apply_thorough_preset(cfg: argparse.Namespace) -> argparse.Namespace:
    if not getattr(cfg, "thorough_mode", False):
        return cfg
    if cfg.warmup_epochs == 0:
        cfg.warmup_epochs = 5
    if cfg.warmup_batch_size == 4:
        cfg.warmup_batch_size = 8
    if cfg.meta_steps == 100:
        cfg.meta_steps = 400
    if cfg.meta_batch_size == 2:
        cfg.meta_batch_size = 3
    if cfg.support_size == 8:
        cfg.support_size = 12
    if cfg.query_size == 16:
        cfg.query_size = 24
    if cfg.inner_steps == 1:
        cfg.inner_steps = 2
    if cfg.n_trainable_encoder_blocks == 2:
        cfg.n_trainable_encoder_blocks = 4
    if cfg.n_trainable_decoder_blocks == 2:
        cfg.n_trainable_decoder_blocks = 4
    if not cfg.train_final_layer_norms:
        cfg.train_final_layer_norms = True
    if not cfg.train_lm_head:
        cfg.train_lm_head = True
    if cfg.adapt_epochs == 15:
        cfg.adapt_epochs = 25
    if cfg.constrained_num_beams == 4:
        cfg.constrained_num_beams = 8
    if cfg.meta_eval_every == 0:
        cfg.meta_eval_every = 25
    if cfg.meta_select_metric == "loss":
        cfg.meta_select_metric = "mean_trait_qwk"
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compact seq2seq T5 pure-MAML for trait generation with constrained decoding and dev-QWK selection.")
    p.add_argument("--source_csv", type=str, required=True)
    p.add_argument("--target_csv", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model_name", type=str, default="google/flan-t5-base")
    p.add_argument("--source_sep", type=str, default="auto")
    p.add_argument("--target_sep", type=str, default="auto")
    p.add_argument("--source_prompt_col", type=str, default="essay_set")
    p.add_argument("--target_prompt_col", type=str, default="essay_set")
    p.add_argument("--source_essay_col", type=str, default="essay")
    p.add_argument("--target_essay_col", type=str, default="essay")
    p.add_argument("--source_prompt_text_col", type=str, default=None)
    p.add_argument("--target_prompt_text_col", type=str, default=None)
    p.add_argument("--source_trait_map", type=str, default=None)
    p.add_argument("--target_trait_map", type=str, default=None)
    p.add_argument("--source_prompt_map", type=str, default=None)
    p.add_argument("--target_prompt_map", type=str, default=None)
    p.add_argument("--canonical_traits", nargs="+", default=CANONICAL_TRAITS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_input_length", type=int, default=512)
    p.add_argument("--max_target_length", type=int, default=24)
    p.add_argument("--generation_max_new_tokens", type=int, default=16)
    p.add_argument("--warmup_epochs", type=int, default=0)
    p.add_argument("--warmup_dev_frac", type=float, default=0.1)
    p.add_argument("--warmup_lr", type=float, default=2e-4)
    p.add_argument("--warmup_batch_size", type=int, default=4)
    p.add_argument("--meta_steps", type=int, default=100)
    p.add_argument("--meta_batch_size", type=int, default=2)
    p.add_argument("--support_size", type=int, default=8)
    p.add_argument("--query_size", type=int, default=16)
    p.add_argument("--inner_lr", type=float, default=1e-3)
    p.add_argument("--inner_steps", type=int, default=1)
    p.add_argument("--outer_lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--n_trainable_encoder_blocks", type=int, default=2)
    p.add_argument("--n_trainable_decoder_blocks", type=int, default=2)
    p.add_argument("--train_final_layer_norms", action="store_true")
    p.add_argument("--train_lm_head", action="store_true")
    p.add_argument("--support_frac", type=float, default=0.6)
    p.add_argument("--dev_frac", type=float, default=0.2)
    p.add_argument("--adapt_epochs", type=int, default=15)
    p.add_argument("--adapt_lr", type=float, default=5e-5)
    p.add_argument("--adapt_batch_size", type=int, default=4)
    p.add_argument("--score_token_weight", type=float, default=4.0)
    p.add_argument("--separator_token_weight", type=float, default=1.0)
    p.add_argument("--constrained_num_beams", type=int, default=4)
    p.add_argument("--prompt_sampling", type=str, default="balanced", choices=["balanced", "random"])
    p.add_argument("--meta_eval_every", type=int, default=0)
    p.add_argument("--meta_monitor_rows_per_prompt", type=int, default=24)
    p.add_argument("--meta_select_metric", type=str, default="loss", choices=["loss", "mean_trait_qwk"])
    p.add_argument("--debug_mode", action="store_true")
    p.add_argument("--debug_examples", type=int, default=6)
    p.add_argument("--thorough_mode", action="store_true")
    cfg = p.parse_args()
    return apply_thorough_preset(cfg)


def main() -> None:
    cfg = parse_args()
    print("RUNNING UPDATED DEBUG VERSION", flush=True)
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    print("Loading and processing datasets...", flush=True)

    source_prompt_map = normalize_prompt_map_keys(safe_json_load(cfg.source_prompt_map)) or DEFAULT_SOURCE_PROMPT_MAP.copy()
    target_prompt_map = normalize_prompt_map_keys(safe_json_load(cfg.target_prompt_map)) or DEFAULT_TARGET_PROMPT_MAP.copy()

    source_df_raw = read_table_auto(cfg.source_csv, cfg.source_sep)
    target_df_raw = read_table_auto(cfg.target_csv, cfg.target_sep)

    source_schema = build_trait_schema(source_df_raw, cfg.source_trait_map, cfg.canonical_traits)
    target_schema = build_trait_schema(target_df_raw, cfg.target_trait_map, cfg.canonical_traits)

    source_df = materialize_canonical_columns(source_df_raw, source_schema, cfg.source_prompt_col)
    target_df = materialize_canonical_columns(target_df_raw, target_schema, cfg.target_prompt_col)

    source_df = source_df[count_available_traits(source_df, cfg.canonical_traits) > 0].reset_index(drop=True)
    target_df = target_df[count_available_traits(target_df, cfg.canonical_traits) > 0].reset_index(drop=True)
    if len(source_df) == 0:
        raise ValueError("No usable source rows remain after trait mapping/filtering.")
    if len(target_df) == 0:
        raise ValueError("No usable target rows remain after trait mapping/filtering.")

    json_dump(os.path.join(cfg.output_dir, "source_trait_schema.json"), source_schema.dataset_to_canonical)
    json_dump(os.path.join(cfg.output_dir, "target_trait_schema.json"), target_schema.dataset_to_canonical)
    json_dump(os.path.join(cfg.output_dir, "source_prompt_map.json"), source_prompt_map)
    json_dump(os.path.join(cfg.output_dir, "target_prompt_map.json"), target_prompt_map)
    json_dump(os.path.join(cfg.output_dir, "run_config.json"), vars(cfg))
    json_dump(os.path.join(cfg.output_dir, "source_dataset_summary.json"), summarize_dataset(source_df, cfg.source_prompt_col, cfg.canonical_traits, "source"))
    json_dump(os.path.join(cfg.output_dir, "target_dataset_summary.json"), summarize_dataset(target_df, cfg.target_prompt_col, cfg.canonical_traits, "target"))
    print("Source dataset summary:", summarize_dataset(source_df, cfg.source_prompt_col, cfg.canonical_traits, "source"))
    print("Target dataset summary:", summarize_dataset(target_df, cfg.target_prompt_col, cfg.canonical_traits, "target"))

    source_ranges = ScoreRangeStore(cfg.canonical_traits)
    source_ranges.fit_from_hardcoded_schema(
        source_df,
        prompt_col=cfg.source_prompt_col,
        hardcoded_ranges=SOURCE_SCORE_RANGES,
        default_range=TARGET_SCORE_RANGE_DEFAULT,
        fallback_to_data_for_missing=True,
    )
    source_ranges.save(os.path.join(cfg.output_dir, "source_prompt_ranges.json"))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = T5ForConditionalGeneration.from_pretrained(
        cfg.model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    configure_trainable_params(
        model,
        n_trainable_encoder_blocks=cfg.n_trainable_encoder_blocks,
        n_trainable_decoder_blocks=cfg.n_trainable_decoder_blocks,
        train_final_layer_norms=cfg.train_final_layer_norms,
        train_lm_head=cfg.train_lm_head,
    )
    json_dump(os.path.join(cfg.output_dir, "trainable_params.json"), count_trainable_parameters(model))
    print("Trainable parameter summary:", count_trainable_parameters(model))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("cuda available:", torch.cuda.is_available(), flush=True)
    print("device:", device, flush=True)
    print_cuda_mem("after cpu model load")

    decoder_start_token_id = model.config.decoder_start_token_id
    if decoder_start_token_id is None:
        decoder_start_token_id = tokenizer.pad_token_id
    eos_token_id = model.config.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    if decoder_start_token_id is None or eos_token_id is None:
        raise ValueError("Could not determine decoder start or EOS token id for constrained decoding.")
    constraint_factory = ConstraintFactory(tokenizer, decoder_start_token_id, eos_token_id)

    if cfg.warmup_epochs > 0:
        source_labels = overall_labels_for_split(source_df, cfg.canonical_traits)
        tr_idx, dv_idx = safe_train_test_split_indices(len(source_df), cfg.warmup_dev_frac, source_labels, cfg.seed)
        warm_train_df = source_df.iloc[tr_idx].reset_index(drop=True)
        warm_dev_df = source_df.iloc[dv_idx].reset_index(drop=True)
        train_ds = build_dataset(
            warm_train_df, tokenizer, cfg.source_prompt_col, cfg.source_essay_col, cfg.source_prompt_text_col,
            source_ranges, cfg.canonical_traits, cfg.max_input_length, cfg.max_target_length, source_prompt_map
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.warmup_batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_seq2seq_batch(x, tokenizer),
        )
        warm_path = supervised_warmup(
            model=model,
            train_loader=train_loader,
            dev_df=warm_dev_df,
            tokenizer=tokenizer,
            range_store=source_ranges,
            prompt_col=cfg.source_prompt_col,
            essay_col=cfg.source_essay_col,
            prompt_text_col=cfg.source_prompt_text_col,
            canonical_traits=cfg.canonical_traits,
            prompt_map=source_prompt_map,
            schema_label_map={t: t for t in cfg.canonical_traits},
            device=device,
            lr=cfg.warmup_lr,
            weight_decay=cfg.weight_decay,
            epochs=cfg.warmup_epochs,
            output_dir=cfg.output_dir,
            max_input_length=cfg.max_input_length,
            generation_max_new_tokens=cfg.generation_max_new_tokens,
            score_token_weight=cfg.score_token_weight,
            separator_token_weight=cfg.separator_token_weight,
            constrained_num_beams=cfg.constrained_num_beams,
            constraint_factory=constraint_factory,
            model_name=cfg.model_name,
            debug_mode=cfg.debug_mode,
            debug_examples=cfg.debug_examples,
        )
        print_cuda_mem("after warmup before cleanup")
        model.cpu()
        clear_cuda()
        state = torch.load(warm_path, map_location="cpu")
        model.load_state_dict(state)
        del state
        clear_cuda()
        print_cuda_mem("after warmup cleanup")

    source_monitor_df = build_monitor_df(source_df, cfg.source_prompt_col, cfg.meta_monitor_rows_per_prompt, cfg.seed) if cfg.meta_eval_every > 0 or cfg.debug_mode else None

    model.cpu()
    clear_cuda()
    print_cuda_mem("before meta_train_maml call")
    meta_path = meta_train_maml(
        model=model,
        train_df=source_df,
        tokenizer=tokenizer,
        range_store=source_ranges,
        prompt_col=cfg.source_prompt_col,
        essay_col=cfg.source_essay_col,
        prompt_text_col=cfg.source_prompt_text_col,
        canonical_traits=cfg.canonical_traits,
        device=device,
        output_dir=cfg.output_dir,
        meta_steps=cfg.meta_steps,
        meta_batch_size=cfg.meta_batch_size,
        support_size=cfg.support_size,
        query_size=cfg.query_size,
        inner_lr=cfg.inner_lr,
        inner_steps=cfg.inner_steps,
        outer_lr=cfg.outer_lr,
        weight_decay=cfg.weight_decay,
        max_input_length=cfg.max_input_length,
        max_target_length=cfg.max_target_length,
        prompt_map=source_prompt_map,
        score_token_weight=cfg.score_token_weight,
        separator_token_weight=cfg.separator_token_weight,
        model_name=cfg.model_name,
        generation_max_new_tokens=cfg.generation_max_new_tokens,
        constrained_num_beams=cfg.constrained_num_beams,
        constraint_factory=constraint_factory,
        prompt_sampling=cfg.prompt_sampling,
        meta_eval_every=cfg.meta_eval_every,
        source_monitor_df=source_monitor_df,
        meta_select_metric=cfg.meta_select_metric,
        debug_mode=cfg.debug_mode,
        debug_examples=cfg.debug_examples,
    )
    print_cuda_mem("after meta_train_maml before cleanup")
    model.cpu()
    clear_cuda()
    state = torch.load(meta_path, map_location="cpu")
    model.load_state_dict(state)
    del state
    clear_cuda()
    print_cuda_mem("after meta cleanup")
    support_df, dev_df, query_df = split_target_by_prompt(
        target_df,
        prompt_col=cfg.target_prompt_col,
        support_frac=cfg.support_frac,
        dev_frac=cfg.dev_frac,
        canonical_traits=cfg.canonical_traits,
        seed=cfg.seed,
    )
    support_df = support_df.reset_index(drop=True)
    dev_df = dev_df.reset_index(drop=True)
    query_df = query_df.reset_index(drop=True)

    print("Starting target adaptation.")
    model.cpu()
    clear_cuda()
    print_cuda_mem("before target adaptation")
    metrics = adapt_on_target_generation(
        model=model,
        tokenizer=tokenizer,
        support_df=support_df,
        dev_df=dev_df,
        query_df=query_df,
        prompt_col=cfg.target_prompt_col,
        essay_col=cfg.target_essay_col,
        prompt_text_col=cfg.target_prompt_text_col,
        canonical_traits=cfg.canonical_traits,
        schema_label_map=target_schema.canonical_to_dataset,
        device=device,
        output_dir=cfg.output_dir,
        adapt_lr=cfg.adapt_lr,
        adapt_epochs=cfg.adapt_epochs,
        adapt_batch_size=cfg.adapt_batch_size,
        max_input_length=cfg.max_input_length,
        max_target_length=cfg.max_target_length,
        generation_max_new_tokens=cfg.generation_max_new_tokens,
        prompt_map=target_prompt_map,
        train_final_layer_norms=cfg.train_final_layer_norms,
        train_lm_head=cfg.train_lm_head,
        n_trainable_encoder_blocks=cfg.n_trainable_encoder_blocks,
        n_trainable_decoder_blocks=cfg.n_trainable_decoder_blocks,
        score_token_weight=cfg.score_token_weight,
        separator_token_weight=cfg.separator_token_weight,
        constrained_num_beams=cfg.constrained_num_beams,
        constraint_factory=constraint_factory,
        model_name=cfg.model_name,
        debug_mode=cfg.debug_mode,
        debug_examples=cfg.debug_examples,
    )

    print("Final target metrics")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    with open(os.path.join(cfg.output_dir, "final_target_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
