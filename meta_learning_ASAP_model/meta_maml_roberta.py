#!/usr/bin/env python3
"""
Trait-wise cross-prompt AES with:
- RoBERTa backbone
- LoRA adapters
- 5 trait heads: ideas, flow, coherence, vocab, grammar
- MLDG-style meta-training on ASAP source prompts
- Small-data target adaptation on VUW (standard fine-tuning or optional target-side meta adaptation)
- Overall score computed post-hoc as rounded mean of 5 traits

This script is an extension of the MEGA-Score paper setup:
- paper backbone: RoBERTa-base + dropout + single regression head
- paper training: MLDG-style meta-learning with source prompts split into meta-train / meta-test
- paper settings: max length 256, dropout 0.3, MSE loss, AdamW lr=1e-5, 4 epochs, small batches

You can run three stages:
1) meta_train      -> source meta-training on ASAP
2) adapt           -> target adaptation on small VUW data
3) predict         -> prediction/evaluation from an adapted checkpoint

Dependencies:
  pip install torch transformers peft higher pandas scikit-learn numpy tqdm

Example:
  python trait_meta_lora_aes.py meta_train \
      --source_file asap_traits.tsv \
      --output_dir runs/source_meta

  python trait_meta_lora_aes.py adapt \
      --target_train_file vuw_train.tsv \
      --target_dev_file vuw_dev.tsv \
      --target_test_file vuw_test.tsv \
      --source_checkpoint runs/source_meta/best.pt \
      --output_dir runs/vuw_adapt
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

try:
    import higher
except ImportError:
    higher = None

from peft import LoraConfig, TaskType, get_peft_model


TRAITS = ["ideas", "flow", "coherence", "vocab", "grammar"]
SOURCE_TO_TARGET = {
    "content": "ideas",
    "sentence_fluency": "flow",
    "organization": "coherence",
    "word_choice": "vocab",
    "conventions": "grammar",
}

DEFAULT_TARGET_RANGES = {
    "ideas": [1.0, 6.0],
    "flow": [1.0, 6.0],
    "coherence": [1.0, 6.0],
    "vocab": [1.0, 6.0],
    "grammar": [1.0, 6.0],
}

DEFAULT_SOURCE_RANGES = "auto"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_table_auto(path: str) -> pd.DataFrame:
    _, ext = os.path.splitext(path.lower())
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    # fallback: infer delimiter
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
    dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
    return pd.read_csv(path, sep=dialect.delimiter)


def parse_ranges(text: Optional[str], default: Dict[str, List[float]] | str, df: Optional[pd.DataFrame] = None) -> Dict[str, Tuple[float, float]]:
    if text is None:
        if default == "auto":
            if df is None:
                raise ValueError("auto ranges requested but no dataframe provided")
            ranges = {}
            for t in TRAITS:
                vals = pd.to_numeric(df[t], errors="coerce").dropna().values.astype(float)
                if len(vals) == 0:
                    raise ValueError(f"No values found for trait '{t}' to infer range")
                ranges[t] = (float(vals.min()), float(vals.max()))
            return ranges
        return {k: (float(v[0]), float(v[1])) for k, v in default.items()}

    obj = json.loads(text)
    return {k: (float(v[0]), float(v[1])) for k, v in obj.items()}


def clamp_round(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(np.rint(x), lo, hi)


class TraitScaler:
    def __init__(self, ranges: Dict[str, Tuple[float, float]]):
        self.ranges = ranges

    def normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for t in TRAITS:
            lo, hi = self.ranges[t]
            denom = hi - lo
            if denom <= 0:
                raise ValueError(f"Invalid range for trait {t}: {(lo, hi)}")
            out[t] = (pd.to_numeric(out[t], errors="coerce") - lo) / denom
        return out

    def denorm_array(self, arr: np.ndarray) -> np.ndarray:
        out = np.zeros_like(arr, dtype=np.float32)
        for i, t in enumerate(TRAITS):
            lo, hi = self.ranges[t]
            out[:, i] = arr[:, i] * (hi - lo) + lo
        return out

    def clip_norm(self, arr: np.ndarray) -> np.ndarray:
        return np.clip(arr, 0.0, 1.0)

    def to_dict(self) -> Dict[str, List[float]]:
        return {k: [float(v[0]), float(v[1])] for k, v in self.ranges.items()}


@dataclass
class Example:
    essay_id: str
    essay_set: str
    essay: str
    labels: np.ndarray


class EssayDataset(Dataset):
    def __init__(self, examples: Sequence[Example], tokenizer, max_length: int):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        ex = self.examples[idx]
        toks = self.tokenizer(
            ex.essay,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "essay_id": ex.essay_id,
            "essay_set": ex.essay_set,
            "input_ids": toks["input_ids"].squeeze(0),
            "attention_mask": toks["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex.labels, dtype=torch.float32),
        }


class MultiTraitAESModel(nn.Module):
    def __init__(
        self,
        model_name: str = "roberta-base",
        dropout: float = 0.3,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        freeze_base: bool = True,
        use_lora: bool = True,
    ):
        super().__init__()
        base = AutoModel.from_pretrained(model_name)

        if freeze_base:
            for p in base.parameters():
                p.requires_grad = False

        # RoBERTa uses query/key/value in self-attn; include dense for a bit more capacity.
        if use_lora:
            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["query", "key", "value", "dense"],
            )
            self.encoder = get_peft_model(base, lora_cfg)
        else:
            self.encoder = base

        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, len(TRAITS))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        # RoBERTa uses <s> at position 0 analogous to [CLS]
        pooled = last_hidden[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.head(pooled)
        return logits

    def trainable_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]


class ManualInnerStepWrapper(nn.Module):
    """
    Fallback wrapper for when higher is unavailable.
    It does NOT perform true differentiable inner-loop adaptation.
    It instead optimizes a mixed objective over meta-train and meta-test batches.

    Recommended: install 'higher' so the script can run the intended MLDG-style update.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(target)
    diff = (pred - torch.nan_to_num(target, nan=0.0)) ** 2
    diff = diff * mask.float()
    denom = mask.float().sum().clamp(min=1.0)
    return diff.sum() / denom


class InfinitePromptBatcher:
    def __init__(self, examples_by_prompt: Dict[str, List[Example]], batch_size: int, seed: int):
        self.examples_by_prompt = examples_by_prompt
        self.batch_size = batch_size
        self.rng = random.Random(seed)

    def sample_examples(self, prompt_id: str) -> List[Example]:
        pool = self.examples_by_prompt[prompt_id]
        if len(pool) >= self.batch_size:
            idxs = self.rng.sample(range(len(pool)), self.batch_size)
            return [pool[i] for i in idxs]
        return [self.rng.choice(pool) for _ in range(self.batch_size)]


class Collator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: Sequence[Example]) -> Dict[str, torch.Tensor | List[str]]:
        texts = [x.essay for x in examples]
        toks = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = torch.tensor(np.stack([x.labels for x in examples], axis=0), dtype=torch.float32)
        return {
            "essay_id": [x.essay_id for x in examples],
            "essay_set": [x.essay_set for x in examples],
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
            "labels": labels,
        }


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def make_examples(df: pd.DataFrame) -> List[Example]:
    examples = []
    for i, row in df.iterrows():
        essay_id = str(row.get("id", row.get("essay_id", i)))
        essay_set = str(row["essay_set"])
        essay = str(row["essay"])
        labels = np.array([row[t] for t in TRAITS], dtype=np.float32)
        examples.append(Example(essay_id=essay_id, essay_set=essay_set, essay=essay, labels=labels))
    return examples


def standardize_source_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for src, tgt in SOURCE_TO_TARGET.items():
        if src in df.columns and tgt not in df.columns:
            df[tgt] = df[src]
    required = {"essay_set", "essay", *TRAITS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Source file is missing columns: {sorted(missing)}")
    return df


def standardize_target_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = {"essay_set", "essay", *TRAITS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Target file is missing columns: {sorted(missing)}")
    return df

def add_overall_for_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "overall" not in df.columns:
        trait_frame = df[TRAITS].apply(pd.to_numeric, errors="coerce")
        df["overall"] = np.rint(trait_frame.mean(axis=1))
    else:
        df["overall"] = pd.to_numeric(df["overall"], errors="coerce")
        missing_mask = df["overall"].isna()
        if missing_mask.any():
            trait_frame = df.loc[missing_mask, TRAITS].apply(pd.to_numeric, errors="coerce")
            df.loc[missing_mask, "overall"] = np.rint(trait_frame.mean(axis=1))
    return df


def _make_overall_bins(overall: pd.Series, num_bins: int) -> pd.Series:
    overall = pd.to_numeric(overall, errors="coerce")
    if overall.isna().all():
        return pd.Series(["na"] * len(overall), index=overall.index)

    uniq = np.unique(overall.dropna().values)
    if len(uniq) == 1:
        return pd.Series([f"b0"] * len(overall), index=overall.index)

    q = min(max(2, num_bins), len(uniq))
    try:
        ranked = overall.rank(method="first")
        bins = pd.qcut(ranked, q=q, labels=False, duplicates="drop")
    except Exception:
        bins = pd.cut(overall, bins=q, labels=False, include_lowest=True, duplicates="drop")

    bins = bins.astype("Int64").astype(str).replace("<NA>", "na")
    return bins.map(lambda x: f"b{x}")


def make_split_labels(df: pd.DataFrame, strategy: str, num_bins: int) -> Optional[pd.Series]:
    if strategy == "random":
        return None

    prompt = df["essay_set"].astype(str).fillna("UNK")
    if strategy == "prompt":
        return prompt

    if strategy == "prompt_overall":
        df2 = add_overall_for_split(df)
        overall_bins = _make_overall_bins(df2["overall"], num_bins=num_bins)
        return prompt + "__" + overall_bins.astype(str)

    raise ValueError(f"Unknown split strategy: {strategy}")


def _split_frame(
    df: pd.DataFrame,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
    split_strategy: str,
    split_num_bins: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    total = train_ratio + dev_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    if min(train_ratio, dev_ratio, test_ratio) < 0:
        raise ValueError("Split ratios must be non-negative")
    if len(df) < 3:
        raise ValueError("Need at least 3 rows to create train/dev/test splits")

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    strategy_order = []
    for s in [split_strategy, "prompt", "random"]:
        if s not in strategy_order:
            strategy_order.append(s)

    temp_ratio = dev_ratio + test_ratio
    for strategy in strategy_order:
        try:
            strat_labels = make_split_labels(df, strategy, split_num_bins)
            train_df, temp_df = train_test_split(
                df,
                test_size=temp_ratio,
                random_state=seed,
                shuffle=True,
                stratify=strat_labels if strat_labels is not None else None,
            )

            if len(temp_df) == 0:
                dev_df = df.iloc[0:0].copy()
                test_df = df.iloc[0:0].copy()
            elif dev_ratio == 0 and test_ratio > 0:
                dev_df = df.iloc[0:0].copy()
                test_df = temp_df.copy()
            elif test_ratio == 0 and dev_ratio > 0:
                dev_df = temp_df.copy()
                test_df = df.iloc[0:0].copy()
            else:
                rel_test = test_ratio / temp_ratio
                temp_labels = make_split_labels(temp_df, strategy, split_num_bins)
                dev_df, test_df = train_test_split(
                    temp_df,
                    test_size=rel_test,
                    random_state=seed,
                    shuffle=True,
                    stratify=temp_labels if temp_labels is not None else None,
                )

            return (
                train_df.reset_index(drop=True),
                dev_df.reset_index(drop=True),
                test_df.reset_index(drop=True),
                strategy,
            )
        except ValueError as e:
            print(f"Split strategy '{strategy}' could not be applied: {e}")

    raise ValueError("Could not create a fair target split with the requested ratios. Try a larger dataset or simpler ratios.")


def save_split_summary(train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str, used_strategy: str) -> None:
    rows = []
    for name, part in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        row = {"split": name, "n": int(len(part)), "strategy": used_strategy}
        for prompt_id, count in part["essay_set"].astype(str).value_counts().sort_index().items():
            row[f"prompt_{prompt_id}"] = int(count)
        rows.append(row)

    summary_df = pd.DataFrame(rows).fillna(0)
    summary_path = os.path.join(output_dir, "split_summary.tsv")
    summary_df.to_csv(summary_path, sep="	", index=False)
    print("Saved target split summary to:", summary_path)
    print(summary_df.to_string(index=False))


def maybe_build_target_splits(args: argparse.Namespace) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if args.target_file is None:
        train_df = standardize_target_df(read_table_auto(args.target_train_file))
        dev_df = standardize_target_df(read_table_auto(args.target_dev_file)) if args.target_dev_file else None
        test_df = standardize_target_df(read_table_auto(args.target_test_file)) if args.target_test_file else None
        return train_df, dev_df, test_df

    full_df = standardize_target_df(read_table_auto(args.target_file))
    full_df = add_overall_for_split(full_df)

    split_dir = os.path.join(args.output_dir, "splits")
    ensure_dir(split_dir)

    train_df, dev_df, test_df, used_strategy = _split_frame(
        full_df,
        train_ratio=args.target_train_ratio,
        dev_ratio=args.target_dev_ratio,
        test_ratio=args.target_test_ratio,
        seed=args.seed,
        split_strategy=args.split_strategy,
        split_num_bins=args.split_num_bins,
    )

    train_path = os.path.join(split_dir, "target_train.tsv")
    dev_path = os.path.join(split_dir, "target_dev.tsv")
    test_path = os.path.join(split_dir, "target_test.tsv")
    train_df.to_csv(train_path, sep="	", index=False)
    dev_df.to_csv(dev_path, sep="	", index=False)
    test_df.to_csv(test_path, sep="	", index=False)
    save_split_summary(train_df, dev_df, test_df, split_dir, used_strategy=used_strategy)

    print("Target file split inside script:")
    print(f"  full file : {args.target_file}")
    print(f"  train     : {train_path}")
    print(f"  dev       : {dev_path}")
    print(f"  test      : {test_path}")
    print(f"  strategy  : {used_strategy}")

    return train_df, dev_df, test_df


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    scaler: TraitScaler,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    model.eval()
    all_ids: List[str] = []
    all_sets: List[str] = []
    gold_norm: List[np.ndarray] = []
    pred_norm: List[np.ndarray] = []

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        logits = model(batch["input_ids"], batch["attention_mask"])
        pred = logits.detach().cpu().numpy()
        gold = batch["labels"].detach().cpu().numpy()
        pred_norm.append(pred)
        gold_norm.append(gold)
        all_ids.extend(batch["essay_id"])
        all_sets.extend(batch["essay_set"])

    if len(pred_norm) == 0:
        return {}

    pred_norm_arr = np.concatenate(pred_norm, axis=0)
    gold_norm_arr = np.concatenate(gold_norm, axis=0)

    # Keep NaNs in the gold labels so evaluation can mask them out later.
    pred_norm_arr = scaler.clip_norm(pred_norm_arr)
    gold_norm_arr = gold_norm_arr.copy()

    pred_raw = scaler.denorm_array(pred_norm_arr)
    gold_raw = scaler.denorm_array(np.nan_to_num(gold_norm_arr, nan=0.0))

    # Restore NaNs after denormalization so metrics can ignore missing labels.
    gold_raw[np.isnan(gold_norm_arr)] = np.nan

    metrics: Dict[str, float] = {}
    pred_rounded = np.zeros_like(pred_raw)
    gold_rounded = np.full_like(gold_raw, np.nan, dtype=np.float32)

    for i, t in enumerate(TRAITS):
        lo, hi = scaler.ranges[t]

        gold_col = gold_raw[:, i]
        pred_col = pred_raw[:, i]

        valid = np.isfinite(gold_col) & np.isfinite(pred_col)

        pred_rounded[:, i] = clamp_round(np.nan_to_num(pred_col, nan=lo), lo, hi)

        if valid.any():
            gold_rounded[valid, i] = clamp_round(gold_col[valid], lo, hi)

            metrics[f"rmse_{t}"] = float(
                math.sqrt(mean_squared_error(gold_col[valid], pred_col[valid]))
            )

            gold_int = gold_rounded[valid, i].astype(int)
            pred_int = pred_rounded[valid, i].astype(int)

            if len(np.unique(gold_int)) < 2 or len(np.unique(pred_int)) < 2:
                metrics[f"qwk_{t}"] = 1.0 if np.array_equal(gold_int, pred_int) else 0.0
            else:
                metrics[f"qwk_{t}"] = float(
                    cohen_kappa_score(gold_int, pred_int, weights="quadratic")
                )
        else:
            metrics[f"rmse_{t}"] = float("nan")
            metrics[f"qwk_{t}"] = float("nan")

    gold_overall = np.rint(np.nanmean(gold_raw, axis=1))
    pred_overall = np.rint(np.nanmean(pred_raw, axis=1))
    overall_lo = float(np.rint(np.mean([scaler.ranges[t][0] for t in TRAITS])))
    overall_hi = float(np.rint(np.mean([scaler.ranges[t][1] for t in TRAITS])))
    gold_overall = np.clip(gold_overall, overall_lo, overall_hi)
    pred_overall = np.clip(pred_overall, overall_lo, overall_hi)

    overall_valid = np.isfinite(gold_overall) & np.isfinite(pred_overall)
    if overall_valid.any():
        metrics["rmse_overall"] = float(
            math.sqrt(mean_squared_error(gold_overall[overall_valid], pred_overall[overall_valid]))
        )
        gold_overall_int = gold_overall[overall_valid].astype(int)
        pred_overall_int = pred_overall[overall_valid].astype(int)
        if len(np.unique(gold_overall_int)) < 2 or len(np.unique(pred_overall_int)) < 2:
            metrics["qwk_overall"] = 1.0 if np.array_equal(gold_overall_int, pred_overall_int) else 0.0
        else:
            metrics["qwk_overall"] = float(
                cohen_kappa_score(gold_overall_int, pred_overall_int, weights="quadratic")
            )
    else:
        metrics["rmse_overall"] = float("nan")
        metrics["qwk_overall"] = float("nan")

    metrics["qwk_mean_traits"] = float(np.nanmean([metrics[f"qwk_{t}"] for t in TRAITS]))
    metrics["rmse_mean_traits"] = float(np.nanmean([metrics[f"rmse_{t}"] for t in TRAITS]))

    if save_path is not None:
        rows = []
        for idx in range(len(all_ids)):
            row = {
                "essay_id": all_ids[idx],
                "essay_set": all_sets[idx],
            }
            for j, t in enumerate(TRAITS):
                row[f"gold_{t}"] = None if not np.isfinite(gold_raw[idx, j]) else float(gold_raw[idx, j])
                row[f"pred_{t}"] = float(pred_raw[idx, j])
                row[f"pred_{t}_rounded"] = int(pred_rounded[idx, j])
                row[f"gold_{t}_rounded"] = None if not np.isfinite(gold_rounded[idx, j]) else int(gold_rounded[idx, j])
            row["gold_overall"] = None if not np.isfinite(gold_overall[idx]) else int(gold_overall[idx])
            row["pred_overall"] = int(pred_overall[idx])
            rows.append(row)
        pd.DataFrame(rows).to_csv(save_path, sep="\t", index=False)

    return metrics


def save_json(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_checkpoint(
    path: str,
    model: nn.Module,
    tokenizer,
    args: argparse.Namespace,
    scaler: TraitScaler,
    extra: Optional[Dict] = None,
) -> None:
    ckpt = {
        "model_state": model.state_dict(),
        "args": vars(args),
        "trait_ranges": scaler.to_dict(),
        "traits": TRAITS,
        "extra": extra or {},
        "tokenizer_name": args.model_name,
    }
    torch.save(ckpt, path)
    tokenizer.save_pretrained(os.path.join(os.path.dirname(path), "tokenizer"))


def load_checkpoint(path: str, device: torch.device, override_model_name: Optional[str] = None) -> Tuple[nn.Module, Dict, TraitScaler, str]:
    ckpt = torch.load(path, map_location=device)
    saved_args = ckpt["args"]
    model_name = override_model_name or saved_args.get("model_name", "roberta-base")
    model = MultiTraitAESModel(
        model_name=model_name,
        dropout=float(saved_args.get("dropout", 0.3)),
        lora_r=int(saved_args.get("lora_r", 8)),
        lora_alpha=int(saved_args.get("lora_alpha", 16)),
        lora_dropout=float(saved_args.get("lora_dropout", 0.1)),
        freeze_base=bool(saved_args.get("freeze_base", True)),
        use_lora=bool(saved_args.get("use_lora", True)),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    scaler = TraitScaler({k: tuple(v) for k, v in ckpt["trait_ranges"].items()})
    tokenizer_name = ckpt.get("tokenizer_name", model_name)
    return model, ckpt, scaler, tokenizer_name


def print_trainable_params(model: nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / max(total, 1)
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  [trainable] {name} {tuple(param.shape)}")


def split_prompts(prompt_ids: Sequence[str], meta_train_ratio: float, rng: random.Random) -> Tuple[List[str], List[str]]:
    prompt_ids = list(prompt_ids)
    rng.shuffle(prompt_ids)
    n_train = max(1, int(round(len(prompt_ids) * meta_train_ratio)))
    n_train = min(n_train, len(prompt_ids) - 1) if len(prompt_ids) > 1 else 1
    meta_train = prompt_ids[:n_train]
    meta_test = prompt_ids[n_train:] if len(prompt_ids) > 1 else prompt_ids[:]
    if len(meta_test) == 0:
        meta_test = meta_train[-1:]
        meta_train = meta_train[:-1] or meta_train
    return meta_train, meta_test


def sample_prompt_batch(
    batcher: InfinitePromptBatcher,
    collator: Collator,
    prompt_ids: Sequence[str],
) -> Dict[str, torch.Tensor | List[str]]:
    exs: List[Example] = []
    for p in prompt_ids:
        exs.extend(batcher.sample_examples(p))
    return collator(exs)


def train_source_meta(args: argparse.Namespace) -> None:
    if higher is None:
        print("WARNING: package 'higher' is not installed. The script will fall back to a weaker mixed-objective approximation.")
        print("Install it for the intended differentiable meta-update: pip install higher")

    ensure_dir(args.output_dir)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = standardize_source_df(read_table_auto(args.source_file))
    source_ranges = parse_ranges(args.source_trait_ranges, DEFAULT_SOURCE_RANGES, df=df)
    scaler = TraitScaler(source_ranges)
    df = scaler.normalize_df(df)

    examples = make_examples(df)
    examples_by_prompt: Dict[str, List[Example]] = defaultdict(list)
    for ex in examples:
        examples_by_prompt[ex.essay_set].append(ex)

    prompt_ids = sorted(examples_by_prompt.keys())
    if len(prompt_ids) < 2:
        raise ValueError("Need at least 2 source prompts for meta-training")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collator = Collator(tokenizer, args.max_length)
    batcher = InfinitePromptBatcher(examples_by_prompt, batch_size=args.per_prompt_batch_size, seed=args.seed)

    args.use_lora = False
    model = MultiTraitAESModel(
        model_name=args.model_name,
        dropout=args.dropout,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        freeze_base=args.freeze_base,
        use_lora=False,
    ).to(device)

    print_trainable_params(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps,
    )

    best_metric = -1e9
    best_path = os.path.join(args.output_dir, "best.pt")
    history = []
    rng = random.Random(args.seed)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(range(args.steps_per_epoch), desc=f"meta-train epoch {epoch}/{args.epochs}")
        epoch_losses = []

        for step in pbar:
            meta_train_prompts, meta_test_prompts = split_prompts(prompt_ids, args.meta_train_ratio, rng)
            meta_train_batch = move_batch_to_device(sample_prompt_batch(batcher, collator, meta_train_prompts), device)
            meta_test_batch = move_batch_to_device(sample_prompt_batch(batcher, collator, meta_test_prompts), device)

            optimizer.zero_grad()

            if higher is not None:
                inner_opt = torch.optim.SGD(trainable_params, lr=args.inner_lr)
                with higher.innerloop_ctx(
                    model,
                    inner_opt,
                    copy_initial_weights=False,
                    track_higher_grads=not args.first_order,
                ) as (fmodel, diffopt):
                    train_logits = fmodel(meta_train_batch["input_ids"], meta_train_batch["attention_mask"])
                    meta_train_loss = masked_mse(train_logits, meta_train_batch["labels"])
                    diffopt.step(meta_train_loss)
                    test_logits = fmodel(meta_test_batch["input_ids"], meta_test_batch["attention_mask"])
                    meta_test_loss = masked_mse(test_logits, meta_test_batch["labels"])
                    total_loss = meta_train_loss + args.meta_test_weight * meta_test_loss
                    total_loss.backward()
            else:
                train_logits = model(meta_train_batch["input_ids"], meta_train_batch["attention_mask"])
                test_logits = model(meta_test_batch["input_ids"], meta_test_batch["attention_mask"])
                meta_train_loss = masked_mse(train_logits, meta_train_batch["labels"])
                meta_test_loss = masked_mse(test_logits, meta_test_batch["labels"])
                total_loss = meta_train_loss + args.meta_test_weight * meta_test_loss
                total_loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            scheduler.step()

            loss_val = float(total_loss.detach().cpu().item())
            epoch_losses.append(loss_val)
            pbar.set_postfix(
                total=f"{loss_val:.4f}",
                meta_train=f"{float(meta_train_loss.detach().cpu()):.4f}",
                meta_test=f"{float(meta_test_loss.detach().cpu()):.4f}",
                prompts=f"{meta_train_prompts}->{meta_test_prompts}",
            )

        epoch_mean_loss = float(np.mean(epoch_losses))

        # lightweight source-side evaluation: evaluate on all source data
        eval_ds = EssayDataset(examples, tokenizer, args.max_length)
        eval_loader = DataLoader(eval_ds, batch_size=args.eval_batch_size, shuffle=False)
        metrics = evaluate_model(model, eval_loader, device, scaler)
        summary = {
            "epoch": epoch,
            "train_loss": epoch_mean_loss,
            **metrics,
        }
        history.append(summary)
        print(json.dumps(summary, indent=2))

        score = metrics["qwk_mean_traits"]
        if score > best_metric:
            best_metric = score
            save_checkpoint(best_path, model, tokenizer, args, scaler, extra={"history": history})
            print(f"Saved new best checkpoint to {best_path}")

    save_json({"history": history}, os.path.join(args.output_dir, "history.json"))
    save_json({"source_trait_ranges": scaler.to_dict()}, os.path.join(args.output_dir, "source_trait_ranges.json"))
    print(f"Done. Best source checkpoint: {best_path}")


def run_standard_adaptation(
    model: nn.Module,
    tokenizer,
    train_examples: List[Example],
    dev_examples: Optional[List[Example]],
    test_examples: Optional[List[Example]],
    scaler: TraitScaler,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    train_ds = EssayDataset(train_examples, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader = None
    test_loader = None
    if dev_examples is not None:
        dev_loader = DataLoader(EssayDataset(dev_examples, tokenizer, args.max_length), batch_size=args.eval_batch_size, shuffle=False)
    if test_examples is not None:
        test_loader = DataLoader(EssayDataset(test_examples, tokenizer, args.max_length), batch_size=args.eval_batch_size, shuffle=False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.adapt_lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.adapt_epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps,
    )

    best_metric = -1e9
    best_path = os.path.join(args.output_dir, "best.pt")
    history = []

    for epoch in range(1, args.adapt_epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"adapt epoch {epoch}/{args.adapt_epochs}")
        epoch_losses = []
        for batch in pbar:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = masked_mse(logits, batch["labels"])
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            scheduler.step()
            lv = float(loss.detach().cpu())
            epoch_losses.append(lv)
            pbar.set_postfix(loss=f"{lv:.4f}")

        epoch_summary = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
        }

        if dev_loader is not None:
            dev_pred_path = os.path.join(args.output_dir, f"dev_predictions_epoch{epoch}.tsv")
            dev_metrics = evaluate_model(model, dev_loader, device, scaler, save_path=dev_pred_path)
            epoch_summary.update({f"dev_{k}": v for k, v in dev_metrics.items()})
            score = dev_metrics["qwk_mean_traits"]
        elif test_loader is not None:
            # if no dev split exists, use test only as a fallback monitor
            test_metrics = evaluate_model(model, test_loader, device, scaler)
            epoch_summary.update({f"test_{k}": v for k, v in test_metrics.items()})
            score = test_metrics["qwk_mean_traits"]
        else:
            score = -epoch_summary["train_loss"]

        history.append(epoch_summary)
        print(json.dumps(epoch_summary, indent=2))

        if score > best_metric:
            best_metric = score
            save_checkpoint(best_path, model, tokenizer, args, scaler, extra={"history": history})
            print(f"Saved new best adapted checkpoint to {best_path}")

    save_json({"history": history}, os.path.join(args.output_dir, "adapt_history.json"))

    # final test evaluation from best checkpoint if available
    if test_loader is not None and os.path.exists(best_path):
        best_model, _, best_scaler, _ = load_checkpoint(best_path, device)
        best_model.to(device)
        best_model.eval()
        test_pred_path = os.path.join(args.output_dir, "test_predictions.tsv")
        test_metrics = evaluate_model(best_model, test_loader, device, best_scaler, save_path=test_pred_path)
        save_json(test_metrics, os.path.join(args.output_dir, "test_metrics.json"))
        print("Final test metrics from best adapted checkpoint:")
        print(json.dumps(test_metrics, indent=2))


def run_target_meta_adaptation(
    model: nn.Module,
    tokenizer,
    train_examples: List[Example],
    dev_examples: Optional[List[Example]],
    test_examples: Optional[List[Example]],
    scaler: TraitScaler,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if higher is None:
        raise RuntimeError("Target-side meta adaptation requires 'higher'. Install it with: pip install higher")

    examples_by_prompt: Dict[str, List[Example]] = defaultdict(list)
    for ex in train_examples:
        examples_by_prompt[ex.essay_set].append(ex)
    prompt_ids = sorted(examples_by_prompt.keys())
    if len(prompt_ids) < 2:
        raise ValueError("Target-side meta adaptation needs at least 2 target prompts")

    batcher = InfinitePromptBatcher(examples_by_prompt, batch_size=args.per_prompt_batch_size, seed=args.seed + 99)
    collator = Collator(tokenizer, args.max_length)

    dev_loader = None
    test_loader = None
    if dev_examples is not None:
        dev_loader = DataLoader(EssayDataset(dev_examples, tokenizer, args.max_length), batch_size=args.eval_batch_size, shuffle=False)
    if test_examples is not None:
        test_loader = DataLoader(EssayDataset(test_examples, tokenizer, args.max_length), batch_size=args.eval_batch_size, shuffle=False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.adapt_lr, weight_decay=args.weight_decay)
    total_steps = args.adapt_epochs * args.steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps,
    )

    best_metric = -1e9
    best_path = os.path.join(args.output_dir, "best.pt")
    history = []
    rng = random.Random(args.seed + 999)

    for epoch in range(1, args.adapt_epochs + 1):
        model.train()
        pbar = tqdm(range(args.steps_per_epoch), desc=f"target-meta-adapt epoch {epoch}/{args.adapt_epochs}")
        losses = []
        for _ in pbar:
            meta_train_prompts, meta_test_prompts = split_prompts(prompt_ids, args.meta_train_ratio, rng)
            meta_train_batch = move_batch_to_device(sample_prompt_batch(batcher, collator, meta_train_prompts), device)
            meta_test_batch = move_batch_to_device(sample_prompt_batch(batcher, collator, meta_test_prompts), device)

            optimizer.zero_grad()
            inner_opt = torch.optim.SGD(trainable_params, lr=args.inner_lr)
            with higher.innerloop_ctx(
                model,
                inner_opt,
                copy_initial_weights=False,
                track_higher_grads=not args.first_order,
            ) as (fmodel, diffopt):
                train_logits = fmodel(meta_train_batch["input_ids"], meta_train_batch["attention_mask"])
                meta_train_loss = masked_mse(train_logits, meta_train_batch["labels"])
                diffopt.step(meta_train_loss)
                test_logits = fmodel(meta_test_batch["input_ids"], meta_test_batch["attention_mask"])
                meta_test_loss = masked_mse(test_logits, meta_test_batch["labels"])
                total_loss = meta_train_loss + args.meta_test_weight * meta_test_loss
                total_loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            scheduler.step()
            lv = float(total_loss.detach().cpu())
            losses.append(lv)
            pbar.set_postfix(total=f"{lv:.4f}", prompts=f"{meta_train_prompts}->{meta_test_prompts}")

        epoch_summary = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
        }

        if dev_loader is not None:
            dev_pred_path = os.path.join(args.output_dir, f"dev_predictions_epoch{epoch}.tsv")
            dev_metrics = evaluate_model(model, dev_loader, device, scaler, save_path=dev_pred_path)
            epoch_summary.update({f"dev_{k}": v for k, v in dev_metrics.items()})
            score = dev_metrics["qwk_mean_traits"]
        elif test_loader is not None:
            test_metrics = evaluate_model(model, test_loader, device, scaler)
            epoch_summary.update({f"test_{k}": v for k, v in test_metrics.items()})
            score = test_metrics["qwk_mean_traits"]
        else:
            score = -epoch_summary["train_loss"]

        history.append(epoch_summary)
        print(json.dumps(epoch_summary, indent=2))

        if score > best_metric:
            best_metric = score
            save_checkpoint(best_path, model, tokenizer, args, scaler, extra={"history": history})
            print(f"Saved new best target-meta-adapted checkpoint to {best_path}")

    save_json({"history": history}, os.path.join(args.output_dir, "adapt_history.json"))

    if test_loader is not None and os.path.exists(best_path):
        best_model, _, best_scaler, _ = load_checkpoint(best_path, device)
        best_model.to(device)
        best_model.eval()
        test_pred_path = os.path.join(args.output_dir, "test_predictions.tsv")
        test_metrics = evaluate_model(best_model, test_loader, device, best_scaler, save_path=test_pred_path)
        save_json(test_metrics, os.path.join(args.output_dir, "test_metrics.json"))
        print("Final test metrics from best target-meta-adapted checkpoint:")
        print(json.dumps(test_metrics, indent=2))


def adapt_target(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt, source_scaler, tokenizer_name = load_checkpoint(
        args.source_checkpoint,
        device=device,
        override_model_name=args.model_name,
    )

    # Source meta-training is done without LoRA to avoid higher/PEFT MRO issues.
    # Attach LoRA only for target adaptation if the loaded checkpoint does not already have it.
    args.use_lora = True
    if not hasattr(model.encoder, "peft_config"):
        base = model.encoder

        if args.freeze_base:
            for p in base.parameters():
                p.requires_grad = False

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["query", "key", "value", "dense"],
        )
        model.encoder = get_peft_model(base, lora_cfg)

    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # For target adaptation/evaluation, use target rubric ranges.
    train_df, dev_df, test_df = maybe_build_target_splits(args)

    target_ranges = parse_ranges(args.target_trait_ranges, DEFAULT_TARGET_RANGES)
    scaler = TraitScaler(target_ranges)

    train_df = scaler.normalize_df(train_df)
    dev_df = scaler.normalize_df(dev_df) if dev_df is not None else None
    test_df = scaler.normalize_df(test_df) if test_df is not None else None

    train_examples = make_examples(train_df)
    dev_examples = make_examples(dev_df) if dev_df is not None else None
    test_examples = make_examples(test_df) if test_df is not None else None

    print("Loaded source checkpoint from:", args.source_checkpoint)
    print("Source training ranges stored in checkpoint:")
    print(json.dumps(source_scaler.to_dict(), indent=2))
    print("Using target ranges for adaptation/evaluation:")
    print(json.dumps(scaler.to_dict(), indent=2))
    print(f"Target split sizes -> train: {len(train_examples)}, dev: {0 if dev_examples is None else len(dev_examples)}, test: {0 if test_examples is None else len(test_examples)}")
    print_trainable_params(model)

    if args.adapt_mode == "standard":
        run_standard_adaptation(model, tokenizer, train_examples, dev_examples, test_examples, scaler, args, device)
    else:
        run_target_meta_adaptation(model, tokenizer, train_examples, dev_examples, test_examples, scaler, args, device)


def predict_only(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, saved_scaler, tokenizer_name = load_checkpoint(args.checkpoint, device=device, override_model_name=args.model_name)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    df = standardize_target_df(read_table_auto(args.input_file))
    # Use the scaler saved in the checkpoint unless user overrides it.
    if args.target_trait_ranges is not None:
        scaler = TraitScaler(parse_ranges(args.target_trait_ranges, DEFAULT_TARGET_RANGES))
    else:
        scaler = saved_scaler

    df_norm = scaler.normalize_df(df)
    examples = make_examples(df_norm)
    loader = DataLoader(EssayDataset(examples, tokenizer, args.max_length), batch_size=args.eval_batch_size, shuffle=False)
    pred_path = os.path.join(args.output_dir, "predictions.tsv")
    metrics = evaluate_model(model, loader, device, scaler, save_path=pred_path)
    save_json(metrics, os.path.join(args.output_dir, "metrics.json"))
    print(json.dumps(metrics, indent=2))
    print(f"Predictions saved to: {pred_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trait-wise MEGA-Score-style meta-learning + LoRA adaptation")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model_name", type=str, default="roberta-base")
    common.add_argument("--max_length", type=int, default=256)
    common.add_argument("--dropout", type=float, default=0.3)
    common.add_argument("--lora_r", type=int, default=8)
    common.add_argument("--lora_alpha", type=int, default=16)
    common.add_argument("--lora_dropout", type=float, default=0.1)
    common.add_argument("--freeze_base", action="store_true")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--weight_decay", type=float, default=0.01)
    common.add_argument("--grad_clip", type=float, default=1.0)
    common.add_argument("--warmup_ratio", type=float, default=0.1)
    common.add_argument("--eval_batch_size", type=int, default=16)
    common.add_argument("--output_dir", type=str, required=True)

    p1 = sub.add_parser("meta_train", parents=[common])
    p1.add_argument("--source_file", type=str, required=True)
    p1.add_argument("--source_trait_ranges", type=str, default=None,
                    help='JSON like {"ideas":[1,6],...}. Default: auto-infer from source file.')
    p1.add_argument("--epochs", type=int, default=4)
    p1.add_argument("--steps_per_epoch", type=int, default=250)
    p1.add_argument("--per_prompt_batch_size", type=int, default=8,
                    help="Paper-style setting samples 8 essays per prompt per subset.")
    p1.add_argument("--lr", type=float, default=1e-5)
    p1.add_argument("--inner_lr", type=float, default=1e-3)
    p1.add_argument("--meta_train_ratio", type=float, default=0.75)
    p1.add_argument("--meta_test_weight", type=float, default=1.0)
    p1.add_argument("--first_order", action="store_true",
                    help="Use first-order approximation in higher inner loop.")

    p2 = sub.add_parser("adapt", parents=[common])
    p2.add_argument("--source_checkpoint", type=str, required=True)
    p2.add_argument("--target_file", type=str, default=None,
                    help="Single labeled target file. If provided, the script will create fair internal train/dev/test splits.")
    p2.add_argument("--target_train_file", type=str, default=None)
    p2.add_argument("--target_dev_file", type=str, default=None)
    p2.add_argument("--target_test_file", type=str, default=None)
    p2.add_argument("--target_train_ratio", type=float, default=0.6)
    p2.add_argument("--target_dev_ratio", type=float, default=0.2)
    p2.add_argument("--target_test_ratio", type=float, default=0.2)
    p2.add_argument("--split_strategy", type=str, default="prompt_overall", choices=["prompt_overall", "prompt", "random"],
                    help="How to split a single target file fairly. prompt_overall stratifies by prompt plus overall-score bins.")
    p2.add_argument("--split_num_bins", type=int, default=3,
                    help="Number of overall-score bins used when split_strategy=prompt_overall.")
    p2.add_argument("--target_trait_ranges", type=str, default=None,
                    help='JSON like {"ideas":[1,6],...}. Default: all traits 1..6.')
    p2.add_argument("--adapt_epochs", type=int, default=20)
    p2.add_argument("--batch_size", type=int, default=8)
    p2.add_argument("--adapt_lr", type=float, default=5e-4)
    p2.add_argument("--adapt_mode", type=str, default="standard", choices=["standard", "meta"],
                    help="standard = recommended few-shot LoRA/head fine-tuning; meta = optional target-side episodic adaptation.")
    p2.add_argument("--steps_per_epoch", type=int, default=50,
                    help="Used only for adapt_mode=meta")
    p2.add_argument("--per_prompt_batch_size", type=int, default=4,
                    help="Used only for adapt_mode=meta")
    p2.add_argument("--inner_lr", type=float, default=1e-3,
                    help="Used only for adapt_mode=meta")
    p2.add_argument("--meta_train_ratio", type=float, default=0.5,
                    help="Used only for adapt_mode=meta")
    p2.add_argument("--meta_test_weight", type=float, default=1.0,
                    help="Used only for adapt_mode=meta")
    p2.add_argument("--first_order", action="store_true",
                    help="Used only for adapt_mode=meta")

    p3 = sub.add_parser("predict", parents=[common])
    p3.add_argument("--checkpoint", type=str, required=True)
    p3.add_argument("--input_file", type=str, required=True)
    p3.add_argument("--target_trait_ranges", type=str, default=None)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "adapt":
        using_single_file = args.target_file is not None
        using_explicit_splits = args.target_train_file is not None
        if not using_single_file and not using_explicit_splits:
            parser.error("For adapt, provide either --target_file or --target_train_file.")
        if using_single_file and using_explicit_splits:
            parser.error("For adapt, use either --target_file or explicit split files, not both.")

    if args.command == "meta_train":
        train_source_meta(args)
    elif args.command == "adapt":
        adapt_target(args)
    elif args.command == "predict":
        predict_only(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
