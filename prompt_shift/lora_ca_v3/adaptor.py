#!/usr/bin/env python3
"""
Rubric-aware LoRA adaptor for cross-prompt AES.

Architecture: Option A — shared regressor applied trait-by-trait.
  - Essay encoder: the same base encoder used in your base checkpoint, with LoRA applied.
  - Rubric encoder: a separate frozen encoder that reads prompt-trait rubric text from asap_prompt_meta_v3.json.
  - Cross-attention: configurable essay→rubric or rubric→essay attention.
  - Fusion before regression: for each trait, essay_pooled is fused with rubric-guided context.
  - Output: a shared scalar regressor is applied to each fused trait representation.

This is different from a residual-correction design. The rubric cross-attention directly changes
the representation that goes into the regressor.
"""

import os
import math
import json
import argparse
import copy
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import cohen_kappa_score, mean_squared_error

import sys
PARENT_DIR = str(Path(__file__).resolve().parents[1])
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from utils import (  # noqa: E402
    TRAIT_COLUMNS,
    ensure_dir,
    save_json,
    load_json,
    set_seed,
    normalize_prompt_id,
    parse_prompt_list,
    parse_int_list,
    build_prompt_text_map,
    build_score_ranges_from_hardcoded,
    build_global_trait_fallback,
    masked_regression_loss,
    format_metrics_for_print,
    load_base_checkpoint_into_model,
    apply_lora_to_encoder,
    count_parameters,
    amp_context,
    get_range_for_trait,
    normalize_score,
    denormalize_score,
    round_to_step,
    masked_mean_pool,
)


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rubric-aware cross-attention LoRA adaptation with trait-wise regressor-input fusion"
    )

    parser.add_argument("--data_path", type=str, required=True, help="Full original dataset; used for fallback ranges")
    parser.add_argument("--split_root", type=str, required=True, help="Root dir created by create_target_fewshot_splits.py")
    parser.add_argument("--base_root", type=str, required=True, help="Root containing base checkpoints")
    parser.add_argument("--base_ckpt_prefix", type=str, default="base_prompt", help="Checkpoint folder prefix, e.g. base_prompt or rubric_fusion_base_prompt")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--prompt_meta_json", type=str, required=True, help="Path to asap_prompt_meta_v3.json")

    parser.add_argument("--sep", type=str, default="\t")
    parser.add_argument("--prompt_col", type=str, default="essay_set")
    parser.add_argument("--text_col", type=str, default="essay")
    parser.add_argument("--id_col", type=str, default="essay_id")

    parser.add_argument("--heldout_prompts", type=str, default="all")
    parser.add_argument("--fewshot_sizes", type=str, default="8,16,32,64,128")
    parser.add_argument(
        "--dev_file_template",
        type=str,
        default="dev_{k}.tsv",
        help=(
            "Template for the k-specific validation file inside each repeat dir. "
            "Example: dev_{k}.tsv makes k=16 use dev_16.tsv. "
            "Use dev.tsv only if you intentionally want the old shared dev set."
        ),
    )

    parser.add_argument("--max_length", type=int, default=480, help="Essay encoder max length")
    parser.add_argument("--rubric_max_length", type=int, default=256, help="Rubric encoder max length")
    parser.add_argument("--rubric_model_name", type=str, default="roberta-base")
    parser.add_argument("--rubric_text_field", type=str, default="cross_attention_score_texts")
    parser.add_argument("--include_source_excerpt", action="store_true", help="Append a short source excerpt to rubric text for source-dependent prompts")
    parser.add_argument("--source_excerpt_chars", type=int, default=1200, help="Characters to append when --include_source_excerpt is enabled")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--cross_attn_lr", type=float, default=-1.0, help="Use -1 to reuse --lr")
    parser.add_argument("--regressor_lr", type=float, default=-1.0, help="Use -1 to reuse --lr")
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

    # Match the latest contrastive base script: essay encoder sees prompt ID/genre
    # by default, while rubric/score texts use metadata score-level directions.
    parser.add_argument(
        "--exclude_prompt_id_in_essay_encoder",
        action="store_true",
        help="Ablation: do not prepend Prompt ID/genre to the essay encoder prompt text.",
    )
    parser.add_argument("--use_rubric_score_contrastive", action="store_true")
    parser.add_argument("--contrastive_weight", type=float, default=0.1)
    parser.add_argument("--contrastive_temperature", type=float, default=0.07)
    parser.add_argument(
        "--contrastive_loss_type",
        type=str,
        default="hard_ce",
        choices=["hard_ce", "soft_ordinal"],
        help=(
            "Type of rubric-score contrastive loss. "
            "hard_ce uses the original one-hot correct score target. "
            "soft_ordinal uses distance-aware soft targets over ordered score levels."
        ),
    )
    parser.add_argument(
        "--contrastive_soft_sigma",
        type=float,
        default=1.0,
        help="Sigma for soft_ordinal contrastive targets. Smaller values make targets sharper.",
    )
    parser.add_argument(
        "--use_contrastive_projection",
        action="store_true",
        help=(
            "Use a small projection head only for the rubric-score contrastive loss. "
            "The projected vectors are used for contrastive similarity, while the original fused "
            "representation still goes to the regression head unchanged."
        ),
    )
    parser.add_argument(
        "--contrastive_projection_dropout",
        type=float,
        default=0.1,
        help="Dropout used inside the optional contrastive projection head.",
    )
    parser.add_argument(
        "--hard_negative_weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the additional hard-negative margin term inside the contrastive loss. "
            "Use 0.0 to disable."
        ),
    )
    parser.add_argument(
        "--hard_negative_margin",
        type=float,
        default=0.1,
        help="Margin used for the hard-negative ranking term.",
    )
    parser.add_argument(
        "--hard_negative_top_k",
        type=int,
        default=1,
        help="Number of highest-scoring wrong rubric candidates used as hard negatives.",
    )
    parser.add_argument(
        "--hard_negative_nearby_only",
        action="store_true",
        help=(
            "When score_candidate_values are available, restrict hard negatives to nearby score "
            "levels first, then fall back to all wrong candidates if no nearby candidate exists."
        ),
    )
    parser.add_argument(
        "--hard_negative_nearby_distance",
        type=float,
        default=1.0,
        help="Maximum raw-score distance used for --hard_negative_nearby_only.",
    )
    parser.add_argument(
        "--score_rubric_text_field",
        type=str,
        default="cross_attention_score_texts",
        help="Metadata key used for score-level rubric candidates in contrastive loss.",
    )

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, default="query,value")
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
    parser.add_argument("--use_rslora", action="store_true", help="Enable rank-stabilized LoRA")
    parser.add_argument("--use_dora", action="store_true", help="Enable DoRA")

    parser.add_argument("--cross_attn_heads", type=int, default=8)
    parser.add_argument(
        "--cross_attn_direction",
        type=str,
        default="auto",
        choices=["auto", "essay_to_rubric", "rubric_to_essay"],
        help=(
            "Direction for the pre-head cross-attention. "
            "auto reads the value from a rubric-aware base checkpoint; "
            "essay_to_rubric uses Q=essay, K/V=rubric; "
            "rubric_to_essay uses Q=rubric, K/V=essay."
        ),
    )
    parser.add_argument("--fusion_init", type=float, default=0.1, help="Initial scale for rubric/trait fusion before the shared regressor")
    parser.add_argument(
        "--pooling_type",
        type=str,
        default="auto",
        choices=["auto", "mean", "attention"],
        help="Use 'auto' to read pooling from a rubric-aware base checkpoint; otherwise force mean/attention pooling.",
    )
    parser.add_argument(
        "--head_rubric_attention",
        action="store_true",
        help="Enable rubric-attentive scoring head. If loading a base checkpoint trained with this head, it is enabled automatically.",
    )
    parser.add_argument(
        "--head_attn_heads",
        type=int,
        default=8,
        help="Number of attention heads inside the rubric-attentive scoring head. Use <=0 to reuse --cross_attn_heads.",
    )
    parser.add_argument("--delta_init", type=float, default=None, help=argparse.SUPPRESS)  # old alias from correction model
    parser.add_argument("--freeze_rubric_encoder", action="store_true", default=True)
    parser.add_argument("--unfreeze_rubric_encoder", action="store_true", help="Optional ablation: train rubric encoder too")
    parser.add_argument("--freeze_shared_regressor", action="store_true", help="Train only LoRA + rubric fusion modules, not the shared trait regressor")
    parser.add_argument("--freeze_base_regressor", action="store_true", help=argparse.SUPPRESS)  # old alias

    # Threshold tuning is done after training, using the same k-specific dev set
    # used for early stopping/model selection. The learned thresholds are then
    # applied to the fixed test set.
    parser.add_argument("--disable_threshold_tuning", action="store_true", help="Disable post-training threshold tuning.")
    parser.add_argument("--threshold_grid_size", type=int, default=81, help="Number of fixed-grid threshold candidates.")
    parser.add_argument("--threshold_max_coord_iters", type=int, default=30, help="Coordinate-search iterations per restart.")
    parser.add_argument("--threshold_random_restarts", type=int, default=20, help="Random/jittered restarts for threshold search.")

    return parser.parse_args()


# ---------------------------------------------------------------------
# Rubric metadata helpers
# ---------------------------------------------------------------------

def load_prompt_meta(path: str) -> Dict[str, Any]:
    meta = load_json(path)
    if "prompts" in meta:
        return {normalize_prompt_id(k): v for k, v in meta["prompts"].items()}
    return {normalize_prompt_id(k): v for k, v in meta.items()}


def normalize_trait_name_for_text(trait: str) -> str:
    return trait.replace("_", " ")


def build_fallback_rubric_text(prompt_id: str, trait: str, prompt_text_map: Dict[str, str]) -> str:
    prompt_text = prompt_text_map.get(prompt_id, f"Prompt {prompt_id}")
    trait_text = normalize_trait_name_for_text(trait)
    return (
        f"Prompt {prompt_id}. Task: {prompt_text} Trait: {trait_text}. "
        f"Use this trait definition and the score range to judge the essay."
    )


def build_essay_prompt_encoder_text(
    prompt_meta: Dict[str, Any],
    prompt_id: str,
    prompt_text_map: Dict[str, str],
    include_prompt_id: bool = True,
) -> str:
    """Build the text paired with the essay in the essay encoder."""
    pmeta = prompt_meta.get(prompt_id, {})
    description = pmeta.get("description") or prompt_text_map.get(prompt_id, f"Prompt {prompt_id}")
    genre = pmeta.get("genre", "")

    parts = []
    if include_prompt_id:
        parts.append(f"Prompt ID: {prompt_id}.")
    if genre:
        parts.append(f"Genre: {genre}.")
    parts.append(f"Task description: {description}")
    return " ".join(parts)


def _safe_sort_score_items(score_texts: Dict[str, Any]) -> List[Any]:
    def _key(kv):
        try:
            return float(kv[0])
        except Exception:
            return str(kv[0])
    return sorted(score_texts.items(), key=_key)


def get_score_level_rubric_items(
    prompt_meta: Dict[str, Any],
    prompt_id: str,
    trait: str,
    prompt_text_map: Dict[str, str],
    score_rubric_text_field: str = "cross_attention_score_texts",
    include_source_excerpt: bool = False,
    source_excerpt_chars: int = 1200,
) -> List[Any]:
    """Return [(score_value, text), ...] for a prompt-trait score rubric."""
    pmeta = prompt_meta.get(prompt_id, {})
    trait_text = normalize_trait_name_for_text(trait)
    genre = pmeta.get("genre", "")

    score_texts = {}
    field_value = pmeta.get(score_rubric_text_field, {})
    if isinstance(field_value, dict):
        trait_value = field_value.get(trait, {})
        if isinstance(trait_value, dict):
            score_texts = trait_value

    if not score_texts and isinstance(pmeta.get("cross_attention_score_texts"), dict):
        trait_value = pmeta["cross_attention_score_texts"].get(trait, {})
        if isinstance(trait_value, dict):
            score_texts = trait_value

    if not score_texts and isinstance(pmeta.get("trait_score_level_directions"), dict):
        trait_value = pmeta["trait_score_level_directions"].get(trait, {})
        if isinstance(trait_value, dict):
            score_texts = trait_value

    items = []
    for score, text in _safe_sort_score_items(score_texts):
        try:
            score_value = float(score)
        except Exception:
            continue

        text = str(text)
        if "direction=" not in text and "score=" not in text:
            text = (
                f"Prompt {prompt_id}; genre={genre}; trait={trait_text}; "
                f"score={score}; direction={text}"
            )

        source = pmeta.get("source_material", {})
        if include_source_excerpt and isinstance(source, dict) and source.get("available"):
            source_title = source.get("title", "")
            source_text = source.get("text", "") or source.get("encoder_text", "")
            if source_text:
                excerpt = source_text[: max(0, source_excerpt_chars)]
                text = f"{text} Source title: {source_title}. Source excerpt: {excerpt}"

        items.append((score_value, text))

    return items


def score_to_candidate_index(
    raw_score: float,
    raw_min: float,
    raw_max: float,
    candidate_scores: List[float],
) -> int:
    """Map a raw dataset score to the closest score-level rubric candidate."""
    if not candidate_scores:
        return -100

    scores = np.asarray(candidate_scores, dtype=np.float32)
    raw_score = float(raw_score)

    direct = np.where(np.isclose(scores, raw_score, atol=1e-4))[0]
    if len(direct) > 0:
        return int(direct[0])

    cand_min, cand_max = float(np.min(scores)), float(np.max(scores))
    mapped = raw_score
    if raw_max > raw_min and cand_max > cand_min:
        norm = (raw_score - raw_min) / (raw_max - raw_min)
        mapped = cand_min + norm * (cand_max - cand_min)

    return int(np.argmin(np.abs(scores - mapped)))


def get_rubric_text(
    prompt_meta: Dict[str, Any],
    prompt_id: str,
    trait: str,
    prompt_text_map: Dict[str, str],
    rubric_text_field: str = "trait_rubric_encoder_text",
    include_source_excerpt: bool = False,
    source_excerpt_chars: int = 1200,
) -> str:
    """Return a single encoder-ready text for a prompt-trait pair."""
    pmeta = prompt_meta.get(prompt_id, {})
    trait_text = normalize_trait_name_for_text(trait)

    text = ""
    if isinstance(pmeta.get(rubric_text_field), dict):
        field_value = pmeta[rubric_text_field].get(trait, "")
        if isinstance(field_value, dict):
            ordered_items = _safe_sort_score_items(field_value)
            joined = " ".join([str(direction) for _, direction in ordered_items])
            genre = pmeta.get("genre", "")
            text = f"Prompt {prompt_id}. Genre: {genre}. Trait: {trait_text}. Score-level directions: {joined}"
        else:
            text = str(field_value) if field_value else ""

    # Fallback 1: score-level texts.
    if not text and isinstance(pmeta.get("cross_attention_score_texts"), dict):
        score_texts = pmeta["cross_attention_score_texts"].get(trait, {})
        if isinstance(score_texts, dict) and score_texts:
            ordered_items = _safe_sort_score_items(score_texts)
            joined = " ".join([str(direction) for _, direction in ordered_items])
            genre = pmeta.get("genre", "")
            text = f"Prompt {prompt_id}. Genre: {genre}. Trait: {trait_text}. Score-level directions: {joined}"

    # Fallback 2: raw level directions.
    if not text and isinstance(pmeta.get("trait_score_level_directions"), dict):
        directions = pmeta["trait_score_level_directions"].get(trait, {})
        if isinstance(directions, dict) and directions:
            ordered_items = sorted(directions.items(), key=lambda kv: float(kv[0]))
            joined = " ".join([f"Score {score}: {direction}" for score, direction in ordered_items])
            genre = pmeta.get("genre", "")
            text = f"Prompt {prompt_id}. Genre: {genre}. Trait: {trait_text}. Score-level directions: {joined}"

    if not text:
        text = build_fallback_rubric_text(prompt_id, trait, prompt_text_map)

    # Optional source excerpt. Keep disabled by default to avoid long-context issues.
    source = pmeta.get("source_material", {})
    if include_source_excerpt and isinstance(source, dict) and source.get("available"):
        source_title = source.get("title", "")
        source_text = source.get("text", "") or source.get("encoder_text", "")
        if source_text:
            excerpt = source_text[: max(0, source_excerpt_chars)]
            text = f"{text} Source title: {source_title}. Source excerpt: {excerpt}"

    return text


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class RubricAwareAESDataset(Dataset):
    """AES dataset returning essay inputs plus prompt-trait rubric inputs.

    For each row:
      input_ids: [essay_max_length]
      attention_mask: [essay_max_length]
      rubric_input_ids: [num_traits, rubric_max_length]
      rubric_attention_mask: [num_traits, rubric_max_length]
      labels: [num_traits]
      label_mask: [num_traits]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        essay_tokenizer,
        rubric_tokenizer,
        trait_cols: List[str],
        prompt_col: str,
        text_col: str,
        prompt_text_map: Dict[str, str],
        score_ranges: Dict[str, Dict[str, Dict[str, float]]],
        global_trait_fallback: Dict[str, Dict[str, float]],
        max_length: int,
        rubric_max_length: int,
        prompt_meta: Dict[str, Any],
        rubric_text_field: str,
        include_source_excerpt: bool,
        source_excerpt_chars: int,
        include_prompt_id_in_essay_encoder: bool = True,
        score_rubric_text_field: str = "cross_attention_score_texts",
    ):
        self.df = df.reset_index(drop=True).copy()
        self.trait_cols = trait_cols
        self.prompt_col = prompt_col
        self.text_col = text_col
        self.score_ranges = score_ranges
        self.global_trait_fallback = global_trait_fallback
        self.prompt_meta = prompt_meta

        self.prompt_ids = [normalize_prompt_id(x) for x in self.df[prompt_col].tolist()]
        prompt_texts = [
            build_essay_prompt_encoder_text(
                prompt_meta=prompt_meta,
                prompt_id=pid,
                prompt_text_map=prompt_text_map,
                include_prompt_id=include_prompt_id_in_essay_encoder,
            )
            for pid in self.prompt_ids
        ]
        essay_texts = self.df[text_col].fillna("").astype(str).tolist()

        enc = essay_tokenizer(
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

        unique_prompt_ids = sorted(set(self.prompt_ids))

        # Score-level rubric candidates for the auxiliary contrastive loss.
        self.score_rubric_items_by_prompt_trait: Dict[str, Dict[str, List[Any]]] = {}
        self.score_candidate_values: Dict[str, Dict[str, List[float]]] = {}
        max_score_candidates = 1
        for pid in unique_prompt_ids:
            self.score_rubric_items_by_prompt_trait[pid] = {}
            self.score_candidate_values[pid] = {}
            for trait in trait_cols:
                items = get_score_level_rubric_items(
                    prompt_meta=prompt_meta,
                    prompt_id=pid,
                    trait=trait,
                    prompt_text_map=prompt_text_map,
                    score_rubric_text_field=score_rubric_text_field,
                    include_source_excerpt=include_source_excerpt,
                    source_excerpt_chars=source_excerpt_chars,
                )
                self.score_rubric_items_by_prompt_trait[pid][trait] = items
                self.score_candidate_values[pid][trait] = [float(score) for score, _ in items]
                max_score_candidates = max(max_score_candidates, len(items))
        self.max_score_candidates = int(max_score_candidates)

        # Labels and masks.
        n = len(self.df)
        t = len(trait_cols)
        self.labels_raw = np.full((n, t), np.nan, dtype=np.float32)
        self.labels_norm = np.zeros((n, t), dtype=np.float32)
        self.label_mask = np.zeros((n, t), dtype=np.float32)
        self.score_class_labels = np.full((n, t), -100, dtype=np.int64)

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

                candidate_scores = self.score_candidate_values.get(pid, {}).get(trait, [])
                self.score_class_labels[i, j] = score_to_candidate_index(
                    raw_score=val,
                    raw_min=mn,
                    raw_max=mx,
                    candidate_scores=candidate_scores,
                )

        # Cache rubric tensors per unique prompt. This avoids storing [N,T,L] copies.
        self.prompt_rubric_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self.score_rubric_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        for pid in unique_prompt_ids:
            rubric_texts = [
                get_rubric_text(
                    prompt_meta=prompt_meta,
                    prompt_id=pid,
                    trait=trait,
                    prompt_text_map=prompt_text_map,
                    rubric_text_field=rubric_text_field,
                    include_source_excerpt=include_source_excerpt,
                    source_excerpt_chars=source_excerpt_chars,
                )
                for trait in trait_cols
            ]
            renc = rubric_tokenizer(
                rubric_texts,
                truncation=True,
                padding="max_length",
                max_length=rubric_max_length,
                return_tensors="pt",
            )
            self.prompt_rubric_cache[pid] = {
                "input_ids": renc["input_ids"],              # [T, R]
                "attention_mask": renc["attention_mask"],    # [T, R]
            }

            # Score-level rubric candidates. Shape: [T, C, R].
            score_texts = []
            candidate_mask = torch.zeros(len(trait_cols), self.max_score_candidates, dtype=torch.float32)
            candidate_values = torch.full((len(trait_cols), self.max_score_candidates), float("nan"), dtype=torch.float32)
            for j, trait in enumerate(trait_cols):
                items = self.score_rubric_items_by_prompt_trait.get(pid, {}).get(trait, [])
                for c in range(self.max_score_candidates):
                    if c < len(items):
                        score_value, text = items[c]
                        score_texts.append(text)
                        candidate_mask[j, c] = 1.0
                        candidate_values[j, c] = float(score_value)
                    else:
                        score_texts.append("")

            srenc = rubric_tokenizer(
                score_texts,
                truncation=True,
                padding="max_length",
                max_length=rubric_max_length,
                return_tensors="pt",
            )
            self.score_rubric_cache[pid] = {
                "input_ids": srenc["input_ids"].reshape(len(trait_cols), self.max_score_candidates, rubric_max_length),
                "attention_mask": srenc["attention_mask"].reshape(len(trait_cols), self.max_score_candidates, rubric_max_length),
                "candidate_mask": candidate_mask,
                "candidate_values": candidate_values,
            }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pid = self.prompt_ids[idx]
        cached = self.prompt_rubric_cache[pid]
        score_cached = self.score_rubric_cache[pid]
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "rubric_input_ids": cached["input_ids"],
            "rubric_attention_mask": cached["attention_mask"],
            "score_rubric_input_ids": score_cached["input_ids"],
            "score_rubric_attention_mask": score_cached["attention_mask"],
            "score_candidate_mask": score_cached["candidate_mask"],
            "score_candidate_values": score_cached["candidate_values"],
            "score_class_labels": torch.tensor(self.score_class_labels[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels_norm[idx], dtype=torch.float32),
            "label_mask": torch.tensor(self.label_mask[idx], dtype=torch.float32),
            "idx": torch.tensor(idx, dtype=torch.long),
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        return item


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class MaskedAttentionPooling(nn.Module):
    """Learned attention pooling over token hidden states.

    Input:  hidden_states [B, L, H], attention_mask [B, L]
    Output: pooled vector [B, H]
    """

    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.scorer(hidden_states).squeeze(-1)
        mask = attention_mask.bool()
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = weights.masked_fill(~mask, 0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)


class SharedTraitRegressor(nn.Module):
    """Shared scalar scorer applied to each trait representation.

    If use_rubric_attention=True, this scoring head attends back to the
    prompt-trait rubric tokens before predicting the scalar score. This makes
    the head itself rubric-attentive, not only the pre-head fusion block.
    """

    def __init__(
        self,
        hidden: int,
        dropout: float,
        old_regressor: Optional[nn.Module] = None,
        use_rubric_attention: bool = False,
        head_attn_heads: int = 8,
    ):
        super().__init__()
        self.use_rubric_attention = use_rubric_attention

        if use_rubric_attention:
            if hidden % head_attn_heads != 0:
                raise ValueError(f"hidden={hidden} must be divisible by head_attn_heads={head_attn_heads}")
            self.head_rubric_attention = nn.MultiheadAttention(
                embed_dim=hidden,
                num_heads=head_attn_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.head_attn_norm = nn.LayerNorm(hidden)
            self.head_ffn_norm = nn.LayerNorm(hidden)
            self.head_dropout = nn.Dropout(dropout)
            self.head_ffn = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
            )

        self.net = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # Best-effort initialization from the old non-rubric base regressor:
        # old regressor = LayerNorm, Dropout, Linear, GELU, Dropout, Linear(hidden, num_traits)
        if isinstance(old_regressor, nn.Sequential) and len(old_regressor) >= 6:
            try:
                self.net[0].load_state_dict(copy.deepcopy(old_regressor[0].state_dict()))
                self.net[2].load_state_dict(copy.deepcopy(old_regressor[2].state_dict()))
                final_layer = old_regressor[-1]
                if isinstance(final_layer, nn.Linear) and final_layer.weight.ndim == 2:
                    with torch.no_grad():
                        self.net[5].weight.copy_(final_layer.weight.mean(dim=0, keepdim=True))
                        self.net[5].bias.copy_(final_layer.bias.mean().view(1))
            except Exception as exc:
                print(f"Warning: could not initialize shared regressor from old regressor: {exc}")

    def forward(
        self,
        x: torch.Tensor,
        rubric_hidden: Optional[torch.Tensor] = None,
        rubric_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_rubric_attention:
            if rubric_hidden is None or rubric_mask is None:
                raise ValueError("rubric_hidden and rubric_mask are required when use_rubric_attention=True")

            # x is one fused trait vector. It becomes a one-token query and
            # attends to rubric tokens for that same prompt-trait pair.
            query = x.unsqueeze(1)  # [B*T, 1, H]
            attn_out, _ = self.head_rubric_attention(
                query=query,
                key=rubric_hidden,
                value=rubric_hidden,
                key_padding_mask=~rubric_mask.bool(),
                need_weights=False,
            )
            attn_out = attn_out.squeeze(1)  # [B*T, H]

            # Residual head-level rubric update.
            x = self.head_attn_norm(x + self.head_dropout(attn_out))

            # Feed-forward refinement inside the head.
            ffn_out = self.head_ffn(x)
            x = self.head_ffn_norm(x + self.head_dropout(ffn_out))

        return self.net(x)


class TraitWiseRegressor(nn.Module):
    """Separate scalar scorer for each trait.

    This matches base checkpoints that save weights under:
        trait_regressors.heads.<trait_idx>.net.*

    It keeps the same fused representation and optional head-level rubric
    attention design, but each trait has its own regression MLP instead of
    sharing one scorer across traits.
    """

    def __init__(
        self,
        hidden: int,
        dropout: float,
        num_traits: int,
        use_rubric_attention: bool = False,
        head_attn_heads: int = 8,
    ):
        super().__init__()
        self.num_traits = num_traits
        self.use_rubric_attention = use_rubric_attention
        self.heads = nn.ModuleList([
            SharedTraitRegressor(
                hidden=hidden,
                dropout=dropout,
                old_regressor=None,
                use_rubric_attention=use_rubric_attention,
                head_attn_heads=head_attn_heads,
            )
            for _ in range(num_traits)
        ])

    def forward(
        self,
        fused: torch.Tensor,
        rubric_hidden: Optional[torch.Tensor] = None,
        rubric_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # fused: [B, T, H]
        if fused.dim() != 3:
            raise ValueError(f"TraitWiseRegressor expects fused [B,T,H], got shape={tuple(fused.shape)}")
        bsz, num_traits, hidden = fused.shape
        if num_traits != self.num_traits:
            raise ValueError(f"fused num_traits={num_traits}, expected={self.num_traits}")

        rubric_hidden_view = None
        rubric_mask_view = None
        if self.use_rubric_attention:
            if rubric_hidden is None or rubric_mask is None:
                raise ValueError("rubric_hidden and rubric_mask are required when use_rubric_attention=True")
            rlen = rubric_hidden.shape[1]
            rubric_hidden_view = rubric_hidden.reshape(bsz, num_traits, rlen, hidden)
            rubric_mask_view = rubric_mask.reshape(bsz, num_traits, rlen)

        outputs = []
        for trait_idx, head in enumerate(self.heads):
            rh = rubric_hidden_view[:, trait_idx, :, :] if rubric_hidden_view is not None else None
            rm = rubric_mask_view[:, trait_idx, :] if rubric_mask_view is not None else None
            outputs.append(head(fused[:, trait_idx, :], rubric_hidden=rh, rubric_mask=rm).squeeze(-1))
        return torch.stack(outputs, dim=1)


class RubricFusionTraitWiseLoRAAESModel(nn.Module):
    """Rubric-aware LoRA AES model with fusion before regression.

    For each trait:
      1. Encode essay with the base essay encoder + LoRA.
      2. Encode the prompt-trait rubric with a frozen rubric encoder.
      3. Use configurable cross-attention direction: essay→rubric or rubric→essay.
      4. Fuse essay_pooled with the rubric-attended context before scoring.
      5. Apply one shared scalar regressor to each fused trait representation.

    This means cross-attention influences the regressor input directly.
    """

    def __init__(
        self,
        base_model: nn.Module,
        rubric_model_name: str,
        num_traits: int,
        dropout: float = 0.1,
        cross_attn_heads: int = 8,
        cross_attn_direction: str = "essay_to_rubric",
        fusion_init: float = 0.1,
        freeze_rubric_encoder: bool = True,
        pooling_type: str = "mean",
        head_rubric_attention: bool = False,
        head_attn_heads: int = 8,
        use_traitwise_regression_heads: bool = False,
        use_contrastive_projection: bool = False,
        contrastive_projection_dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = base_model.encoder
        self.num_traits = num_traits
        self.freeze_rubric_encoder = freeze_rubric_encoder

        hidden = self.encoder.config.hidden_size
        self.hidden_size = hidden
        self.pooling_type = pooling_type
        self.cross_attn_direction = cross_attn_direction
        self.head_rubric_attention = head_rubric_attention
        self.head_attn_heads = head_attn_heads
        self.use_traitwise_regression_heads = use_traitwise_regression_heads
        self.use_contrastive_projection = use_contrastive_projection

        if self.cross_attn_direction not in {"essay_to_rubric", "rubric_to_essay"}:
            raise ValueError(
                f"Unsupported cross_attn_direction={self.cross_attn_direction}. "
                "Use 'essay_to_rubric' or 'rubric_to_essay'."
            )

        if pooling_type == "attention":
            self.essay_pooler = MaskedAttentionPooling(hidden, dropout=dropout)
            self.context_pooler = MaskedAttentionPooling(hidden, dropout=dropout)
        elif pooling_type == "mean":
            self.essay_pooler = None
            self.context_pooler = None
        else:
            raise ValueError(f"Unsupported pooling_type: {pooling_type}")

        self.rubric_encoder = AutoModel.from_pretrained(rubric_model_name)
        rubric_hidden = self.rubric_encoder.config.hidden_size
        self.rubric_projection = nn.Identity() if rubric_hidden == hidden else nn.Linear(rubric_hidden, hidden)

        if hidden % cross_attn_heads != 0:
            raise ValueError(f"hidden_size={hidden} must be divisible by cross_attn_heads={cross_attn_heads}")

        self.trait_embeddings = nn.Embedding(num_traits, hidden)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=cross_attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(hidden)
        self.fusion_dropout = nn.Dropout(dropout)
        self.rubric_fusion_scale = nn.Parameter(torch.tensor(float(fusion_init), dtype=torch.float32))
        self.trait_fusion_scale = nn.Parameter(torch.tensor(float(fusion_init), dtype=torch.float32))

        # Scoring head. Older/base variants used one shared scalar scorer, while
        # newer base checkpoints may use separate trait-wise regression heads.
        # The adaptor auto-selects the correct structure when loading a checkpoint.
        if use_traitwise_regression_heads:
            self.trait_regressors = TraitWiseRegressor(
                hidden=hidden,
                dropout=dropout,
                num_traits=num_traits,
                use_rubric_attention=head_rubric_attention,
                head_attn_heads=head_attn_heads,
            )
            self.shared_regressor = None
        else:
            self.shared_regressor = SharedTraitRegressor(
                hidden=hidden,
                dropout=dropout,
                old_regressor=getattr(base_model, "regressor", None),
                use_rubric_attention=head_rubric_attention,
                head_attn_heads=head_attn_heads,
            )
            self.trait_regressors = None

        # Optional loss-only projection head. It is used only to compute the
        # rubric-score contrastive objective and does not change the regression
        # scorer input/output path.
        if self.use_contrastive_projection:
            self.contrastive_projection = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Dropout(float(contrastive_projection_dropout)),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(float(contrastive_projection_dropout)),
                nn.Linear(hidden, hidden),
            )
        else:
            self.contrastive_projection = None

        if self.freeze_rubric_encoder:
            for p in self.rubric_encoder.parameters():
                p.requires_grad = False

    def pool_tokens(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, which: str) -> torch.Tensor:
        if self.pooling_type == "attention":
            if which == "essay":
                return self.essay_pooler(hidden_states, attention_mask)
            if which == "context":
                return self.context_pooler(hidden_states, attention_mask)
            raise ValueError(f"Unknown pooling target: {which}")
        return masked_mean_pool(hidden_states, attention_mask)

    def encode_rubric(self, rubric_input_ids: torch.Tensor, rubric_attention_mask: torch.Tensor) -> torch.Tensor:
        # rubric_input_ids: [B, T, L]
        bsz, num_traits, rlen = rubric_input_ids.shape
        flat_ids = rubric_input_ids.reshape(bsz * num_traits, rlen)
        flat_mask = rubric_attention_mask.reshape(bsz * num_traits, rlen)

        if self.freeze_rubric_encoder:
            self.rubric_encoder.eval()
            with torch.no_grad():
                out = self.rubric_encoder(input_ids=flat_ids, attention_mask=flat_mask)
                hidden = out.last_hidden_state
        else:
            out = self.rubric_encoder(input_ids=flat_ids, attention_mask=flat_mask)
            hidden = out.last_hidden_state

        hidden = self.rubric_projection(hidden)
        return hidden

    def encode_score_rubrics(
        self,
        score_rubric_input_ids: torch.Tensor,
        score_rubric_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # score_rubric_input_ids: [B, T, C, R]
        bsz, num_traits, num_candidates, rlen = score_rubric_input_ids.shape
        flat_ids = score_rubric_input_ids.reshape(bsz * num_traits * num_candidates, rlen)
        flat_mask = score_rubric_attention_mask.reshape(bsz * num_traits * num_candidates, rlen)

        if self.freeze_rubric_encoder:
            self.rubric_encoder.eval()
            with torch.no_grad():
                out = self.rubric_encoder(input_ids=flat_ids, attention_mask=flat_mask)
                hidden = out.last_hidden_state
        else:
            out = self.rubric_encoder(input_ids=flat_ids, attention_mask=flat_mask)
            hidden = out.last_hidden_state

        hidden = self.rubric_projection(hidden)
        pooled = self.pool_tokens(hidden, flat_mask, which="context")
        return pooled.reshape(bsz, num_traits, num_candidates, self.hidden_size)

    def project_for_contrastive(
        self,
        fused: torch.Tensor,
        score_rubric_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project vectors for contrastive loss only.

        The original fused representation is still used by the regression head.
        This method only creates a separate metric-learning space for the
        rubric-score contrastive objective.
        """
        if self.contrastive_projection is None:
            return fused, score_rubric_emb

        bsz, num_traits, hidden = fused.shape
        num_candidates = score_rubric_emb.shape[2]
        fused_proj = self.contrastive_projection(fused.reshape(bsz * num_traits, hidden))
        score_proj = self.contrastive_projection(
            score_rubric_emb.reshape(bsz * num_traits * num_candidates, hidden)
        )
        fused_proj = fused_proj.reshape(bsz, num_traits, hidden)
        score_proj = score_proj.reshape(bsz, num_traits, num_candidates, hidden)
        return fused_proj, score_proj

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rubric_input_ids: Optional[torch.Tensor] = None,
        rubric_attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_fused: bool = False,
    ) -> torch.Tensor:
        essay_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        essay_hidden = essay_out.last_hidden_state  # [B, S, H]
        essay_pooled = self.pool_tokens(essay_hidden, attention_mask, which="essay")  # [B, H]

        bsz, seq_len, hidden = essay_hidden.shape

        if rubric_input_ids is None or rubric_attention_mask is None:
            # Fallback: apply shared trait scorer to essay representation plus trait embeddings.
            trait_ids = torch.arange(self.num_traits, device=input_ids.device).unsqueeze(0).expand(bsz, self.num_traits)
            trait_vec = self.trait_embeddings(trait_ids)
            essay_rep = essay_pooled.unsqueeze(1).expand(bsz, self.num_traits, hidden)
            fused = self.fusion_norm(essay_rep + self.trait_fusion_scale * trait_vec)
            fused_flat = fused.reshape(bsz * self.num_traits, hidden)
            if self.head_rubric_attention:
                raise ValueError("rubric tensors are required because head_rubric_attention=True")
            if self.use_traitwise_regression_heads:
                preds = self.trait_regressors(fused)
            else:
                preds = self.shared_regressor(fused_flat).reshape(bsz, self.num_traits)
            if return_fused:
                return {"preds": preds, "fused": fused}
            return preds

        num_traits = rubric_input_ids.shape[1]
        if num_traits != self.num_traits:
            raise ValueError(f"rubric num_traits={num_traits}, expected={self.num_traits}")

        rubric_hidden = self.encode_rubric(rubric_input_ids, rubric_attention_mask)  # [B*T, R, H]
        rubric_mask = rubric_attention_mask.reshape(bsz * num_traits, rubric_attention_mask.shape[-1]).bool()

        # Repeat essay hidden states for each trait.
        essay_rep = essay_hidden.unsqueeze(1).expand(bsz, num_traits, seq_len, hidden).reshape(bsz * num_traits, seq_len, hidden)
        essay_mask_rep = attention_mask.unsqueeze(1).expand(bsz, num_traits, seq_len).reshape(bsz * num_traits, seq_len)
        essay_pooled_rep = essay_pooled.unsqueeze(1).expand(bsz, num_traits, hidden).reshape(bsz * num_traits, hidden)

        trait_ids = torch.arange(num_traits, device=input_ids.device).unsqueeze(0).expand(bsz, num_traits).reshape(-1)
        trait_vec = self.trait_embeddings(trait_ids)  # [B*T, H]

        # Configurable pre-head cross-attention direction.
        #
        # essay_to_rubric:
        #   Q = essay tokens, K/V = rubric tokens
        #   Output length = essay_len, so pool with essay_mask_rep.
        #
        # rubric_to_essay:
        #   Q = rubric tokens, K/V = essay tokens
        #   Output length = rubric_len, so pool with rubric_mask.
        if self.cross_attn_direction == "essay_to_rubric":
            query = essay_rep + trait_vec.unsqueeze(1)
            attn_out, _ = self.cross_attention(
                query=query,
                key=rubric_hidden,
                value=rubric_hidden,
                key_padding_mask=~rubric_mask,
                need_weights=False,
            )
            rubric_context = self.pool_tokens(attn_out, essay_mask_rep, which="context")  # [B*T, H]

        elif self.cross_attn_direction == "rubric_to_essay":
            query = rubric_hidden + trait_vec.unsqueeze(1)
            attn_out, _ = self.cross_attention(
                query=query,
                key=essay_rep,
                value=essay_rep,
                key_padding_mask=~essay_mask_rep.bool(),
                need_weights=False,
            )
            rubric_context = self.pool_tokens(attn_out, rubric_mask, which="context")  # [B*T, H]

        else:
            raise ValueError(f"Unsupported cross_attn_direction={self.cross_attn_direction}")

        # Cross-attention influences the regressor input here.
        fused = self.fusion_norm(
            essay_pooled_rep
            + self.rubric_fusion_scale * self.fusion_dropout(rubric_context)
            + self.trait_fusion_scale * trait_vec
        )

        fused_3d = fused.reshape(bsz, num_traits, hidden)
        if self.use_traitwise_regression_heads:
            trait_scores = self.trait_regressors(
                fused_3d,
                rubric_hidden=rubric_hidden if self.head_rubric_attention else None,
                rubric_mask=rubric_mask if self.head_rubric_attention else None,
            )
        else:
            trait_scores = self.shared_regressor(
                fused,
                rubric_hidden=rubric_hidden if self.head_rubric_attention else None,
                rubric_mask=rubric_mask if self.head_rubric_attention else None,
            ).reshape(bsz, num_traits)
        if return_fused:
            return {"preds": trait_scores, "fused": fused_3d}
        return trait_scores


def rubric_score_contrastive_loss(
    fused: torch.Tensor,
    score_rubric_emb: torch.Tensor,
    score_class_labels: torch.Tensor,
    label_mask: torch.Tensor,
    score_candidate_mask: torch.Tensor,
    temperature: float = 0.07,
    loss_type: str = "hard_ce",
    score_candidate_values: Optional[torch.Tensor] = None,
    soft_sigma: float = 1.0,
    hard_negative_weight: float = 0.0,
    hard_negative_margin: float = 0.1,
    hard_negative_top_k: int = 1,
    hard_negative_nearby_only: bool = False,
    hard_negative_nearby_distance: float = 1.0,
) -> torch.Tensor:
    """Contrast fused essay-trait reps against score-level rubric embeddings.

    fused:                 [B, T, H]
    score_rubric_emb:      [B, T, C, H]
    score_class_labels:    [B, T], class index into C or -100 if invalid
    label_mask:            [B, T]
    score_candidate_mask:  [B, T, C]
    score_candidate_values:[B, T, C], raw score value for each candidate

    loss_type:
      - hard_ce: original one-hot cross-entropy target.
      - soft_ordinal: distance-aware soft target distribution. The true score
        candidate receives the highest probability; neighbouring score levels
        receive smaller non-zero probabilities; far score levels are pushed away
        more strongly.
    """
    fused = torch.nn.functional.normalize(fused, dim=-1)
    score_rubric_emb = torch.nn.functional.normalize(score_rubric_emb, dim=-1)

    logits = torch.einsum("bth,btch->btc", fused, score_rubric_emb) / max(float(temperature), 1e-8)

    # autocast may make logits fp16/bf16; masking with -1e9 can overflow in fp16.
    logits = logits.float()
    score_candidate_mask = score_candidate_mask.to(device=logits.device)
    mask_fill_value = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(score_candidate_mask < 0.5, mask_fill_value)

    valid = (label_mask > 0.5) & (score_class_labels >= 0)
    if valid.sum().item() == 0:
        return logits.sum() * 0.0

    def hard_negative_margin_loss() -> torch.Tensor:
        """Extra loss that focuses on high-scoring wrong rubric candidates.

        For each valid essay-trait pair, the positive is the gold score rubric.
        Hard negatives are the wrong rubric candidates with the highest current
        similarity. Optionally, the candidate pool is restricted to nearby score
        levels first, because adjacent score levels are often the hardest AES
        decisions.
        """
        if float(hard_negative_weight) <= 0.0:
            return logits.sum() * 0.0

        top_k = max(1, int(hard_negative_top_k))
        candidate_mask_bool = score_candidate_mask.to(device=logits.device).bool()
        safe_labels = score_class_labels.clamp(min=0).long().to(device=logits.device)
        positive_mask = torch.zeros_like(candidate_mask_bool)
        positive_mask.scatter_(-1, safe_labels.unsqueeze(-1), True)
        negative_mask_all = candidate_mask_bool & (~positive_mask)

        negative_mask = negative_mask_all
        if hard_negative_nearby_only and score_candidate_values is not None:
            candidate_values_hn = score_candidate_values.to(device=logits.device, dtype=logits.dtype)
            true_values_hn = torch.gather(
                candidate_values_hn,
                dim=-1,
                index=safe_labels.unsqueeze(-1),
            ).squeeze(-1)
            candidate_values_hn = torch.nan_to_num(candidate_values_hn, nan=0.0)
            true_values_hn = torch.nan_to_num(true_values_hn, nan=0.0)
            distances_hn = torch.abs(candidate_values_hn - true_values_hn.unsqueeze(-1))
            nearby_mask = (distances_hn > 1e-8) & (distances_hn <= float(hard_negative_nearby_distance))
            nearby_mask = nearby_mask & negative_mask_all

            # If a pair has no nearby negative, fall back to all wrong candidates
            # so the loss remains defined for sparse score ranges.
            has_nearby = nearby_mask.any(dim=-1, keepdim=True)
            negative_mask = torch.where(has_nearby, nearby_mask, negative_mask_all)

        valid_hn = valid & negative_mask.any(dim=-1)
        if valid_hn.sum().item() == 0:
            return logits.sum() * 0.0

        pos_logits = torch.gather(logits, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        neg_logits = logits.masked_fill(~negative_mask, float("-inf"))
        k_eff = min(top_k, neg_logits.shape[-1])
        top_neg_logits = torch.topk(neg_logits, k=k_eff, dim=-1).values

        pair_losses = torch.nn.functional.softplus(
            top_neg_logits - pos_logits.unsqueeze(-1) + float(hard_negative_margin)
        )
        finite_neg = torch.isfinite(top_neg_logits)
        pair_losses = pair_losses.masked_fill(~finite_neg, 0.0)
        denom = finite_neg.float().sum(dim=-1).clamp(min=1.0)
        pair_losses = pair_losses.sum(dim=-1) / denom
        return pair_losses[valid_hn].mean()

    loss_type = str(loss_type).lower()
    if loss_type == "hard_ce":
        logits_valid = logits[valid]
        targets_valid = score_class_labels[valid].long()
        base_loss = torch.nn.functional.cross_entropy(logits_valid, targets_valid)
        return base_loss + float(hard_negative_weight) * hard_negative_margin_loss()

    if loss_type != "soft_ordinal":
        raise ValueError(f"Unsupported contrastive loss_type={loss_type}. Use hard_ce or soft_ordinal.")

    if score_candidate_values is None:
        raise ValueError("score_candidate_values is required when contrastive_loss_type='soft_ordinal'")

    candidate_values = score_candidate_values.to(device=logits.device, dtype=logits.dtype)
    candidate_mask = score_candidate_mask.to(dtype=logits.dtype)

    safe_labels = score_class_labels.clamp(min=0).long().to(device=logits.device)
    true_values = torch.gather(candidate_values, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)

    # Invalid score candidates have NaN values in the dataset cache. Replace the
    # NaNs before distance computation; the mask removes them from the target.
    candidate_values = torch.nan_to_num(candidate_values, nan=0.0)
    true_values = torch.nan_to_num(true_values, nan=0.0)

    sigma = max(float(soft_sigma), 1e-8)
    distances = torch.abs(candidate_values - true_values.unsqueeze(-1))
    soft_targets = torch.exp(-distances / sigma) * candidate_mask
    soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    log_probs = torch.log_softmax(logits, dim=-1)
    per_pair_loss = -(soft_targets * log_probs).sum(dim=-1)
    base_loss = per_pair_loss[valid].mean()
    return base_loss + float(hard_negative_weight) * hard_negative_margin_loss()


# ---------------------------------------------------------------------
# Trainability and optimizer helpers
# ---------------------------------------------------------------------

def mark_only_lora_cross_attention_trainable(model: RubricFusionTraitWiseLoRAAESModel, freeze_shared_regressor: bool = False):
    for p in model.parameters():
        p.requires_grad = False

    # LoRA parameters on essay encoder.
    for name, p in model.encoder.named_parameters():
        if "lora_" in name or "modules_to_save" in name:
            p.requires_grad = True

    # Scoring head: either shared_regressor or trait_regressors depending on
    # the base checkpoint architecture.
    if not freeze_shared_regressor:
        scoring_head = model.trait_regressors if getattr(model, "use_traitwise_regression_heads", False) else model.shared_regressor
        if scoring_head is not None:
            for p in scoring_head.parameters():
                p.requires_grad = True

    # New rubric-aware fusion modules.
    for module in [
        model.rubric_projection,
        model.trait_embeddings,
        model.cross_attention,
        model.fusion_norm,
        model.essay_pooler,
        model.context_pooler,
        getattr(model, "contrastive_projection", None),
    ]:
        if module is None:
            continue
        for p in module.parameters():
            p.requires_grad = True
    model.rubric_fusion_scale.requires_grad = True
    model.trait_fusion_scale.requires_grad = True

    if not model.freeze_rubric_encoder:
        for p in model.rubric_encoder.parameters():
            p.requires_grad = True


def build_optimizer(model: nn.Module, args):
    cross_lr = args.cross_attn_lr if args.cross_attn_lr > 0 else args.lr
    reg_lr = args.regressor_lr if args.regressor_lr > 0 else args.lr

    lora_params = []
    regressor_params = []
    cross_params = []
    rubric_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".lora_" in name or "lora_" in name:
            lora_params.append(p)
        elif name.startswith("shared_regressor.") or name.startswith("trait_regressors."):
            regressor_params.append(p)
        elif name.startswith("rubric_encoder."):
            rubric_params.append(p)
        elif any(name.startswith(prefix) for prefix in [
            "cross_attention.",
            "fusion_norm.",
            "trait_embeddings.",
            "rubric_projection.",
            "essay_pooler.",
            "context_pooler.",
            "contrastive_projection.",
        ]) or name in {"rubric_fusion_scale", "trait_fusion_scale"}:
            cross_params.append(p)
        else:
            other_params.append(p)

    groups = []
    if lora_params:
        groups.append({"params": lora_params, "lr": args.lr, "weight_decay": args.weight_decay})
    if regressor_params:
        groups.append({"params": regressor_params, "lr": reg_lr, "weight_decay": args.weight_decay})
    if cross_params:
        groups.append({"params": cross_params, "lr": cross_lr, "weight_decay": args.weight_decay})
    if rubric_params:
        groups.append({"params": rubric_params, "lr": cross_lr * 0.1, "weight_decay": args.weight_decay})
    if other_params:
        groups.append({"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay})

    if not groups:
        raise RuntimeError("No trainable parameters found.")
    return torch.optim.AdamW(groups)


def trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    state = model.state_dict()
    out = {}
    for key, value in state.items():
        if key in trainable_names:
            out[key] = value.detach().cpu().clone()
    return out


def load_partial_state_dict(model: nn.Module, state: Dict[str, torch.Tensor], device: torch.device):
    state = {k: v.to(device) for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"Warning: unexpected keys while loading best partial state: {unexpected}")
    return missing, unexpected


# ---------------------------------------------------------------------
# Evaluation and prediction
# ---------------------------------------------------------------------

def quadratic_weighted_kappa_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except Exception:
        return float("nan")


def evaluate_rubric_model(
    model: nn.Module,
    dataloader,
    dataset: RubricAwareAESDataset,
    trait_cols: List[str],
    score_ranges: Dict[str, Dict[str, Dict[str, float]]],
    global_trait_fallback: Dict[str, Dict[str, float]],
    device: torch.device,
    round_step: float,
    loss_type: str,
    huber_delta: float,
) -> Dict[str, Any]:
    model.eval()
    all_preds, all_labels, all_masks, all_indices = [], [], [], []
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            rubric_input_ids = batch["rubric_input_ids"].to(device)
            rubric_attention_mask = batch["rubric_attention_mask"].to(device)
            labels = batch["labels"].to(device)
            label_mask = batch["label_mask"].to(device)
            idxs = batch["idx"].cpu().numpy()

            preds = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                rubric_input_ids=rubric_input_ids,
                rubric_attention_mask=rubric_attention_mask,
            )
            loss = masked_regression_loss(preds, labels, label_mask, loss_type=loss_type, huber_delta=huber_delta)
            total_loss += loss.item()
            total_steps += 1

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_masks.append(label_mask.cpu().numpy())
            all_indices.append(idxs)

    if not all_preds:
        return {"loss": float("nan"), "mean_qwk": float("nan"), "mean_rmse": float("nan"), "trait_metrics": {}}

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
            gold_rounded = float(np.clip(round_to_step(gold_raw, round_step), mn, mx))
            pred_rounded = float(np.clip(round_to_step(pred_raw, round_step), mn, mx))
            trait_gold[trait].append(gold_rounded)
            trait_pred[trait].append(pred_rounded)

    trait_metrics = {}
    qwk_values = []
    rmse_values = []
    for trait in trait_cols:
        y_true = np.array(trait_gold[trait], dtype=np.float32)
        y_pred = np.array(trait_pred[trait], dtype=np.float32)
        if len(y_true) == 0:
            trait_metrics[trait] = {"n": 0, "qwk": float("nan"), "rmse": float("nan")}
            continue
        qwk = quadratic_weighted_kappa_safe(y_true, y_pred)
        rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
        if not math.isnan(qwk):
            qwk_values.append(qwk)
        if not math.isnan(rmse):
            rmse_values.append(rmse)
        trait_metrics[trait] = {"n": int(len(y_true)), "qwk": qwk, "rmse": rmse}

    return {
        "loss": total_loss / max(total_steps, 1),
        "mean_qwk": float(np.mean(qwk_values)) if qwk_values else float("nan"),
        "mean_rmse": float(np.mean(rmse_values)) if rmse_values else float("nan"),
        "trait_metrics": trait_metrics,
    }


def predict_to_dataframe_rubric(
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
    model.eval()
    all_preds, all_labels, all_masks, all_indices = [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            rubric_input_ids = batch["rubric_input_ids"].to(device)
            rubric_attention_mask = batch["rubric_attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            label_mask = batch["label_mask"].cpu().numpy()
            idxs = batch["idx"].cpu().numpy()
            preds = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                rubric_input_ids=rubric_input_ids,
                rubric_attention_mask=rubric_attention_mask,
            )
            all_preds.append(preds.cpu().numpy())
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
        row = {"row_idx": ds_idx, prompt_col: prompt_id, text_col: src_row[text_col] if text_col in df_reset.columns else ""}
        if id_col in df_reset.columns:
            row[id_col] = src_row[id_col]

        for j, trait in enumerate(trait_cols):
            if masks[row_pos, j] < 0.5:
                row[f"target_{trait}"] = np.nan
                row[f"pred_{trait}"] = np.nan
                row[f"raw_pred_{trait}"] = np.nan
                continue
            rng = get_range_for_trait(score_ranges, prompt_id, trait, global_trait_fallback)
            mn, mx = rng["min"], rng["max"]
            gold_raw = denormalize_score(labels_norm[row_pos, j], mn, mx)
            pred_raw = denormalize_score(preds_norm[row_pos, j], mn, mx)
            row[f"target_{trait}"] = float(np.clip(round_to_step(gold_raw, round_step), mn, mx))
            row[f"pred_{trait}"] = float(np.clip(round_to_step(pred_raw, round_step), mn, mx))
            row[f"raw_pred_{trait}"] = float(pred_raw)
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
        for prefix in ["target", "pred", "raw_pred"]:
            col = f"{prefix}_{trait}"
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
    labels = mn + np.arange(n, dtype=np.float64) * step
    labels = np.clip(labels, mn, mx)
    return labels


def default_midpoint_thresholds(labels: np.ndarray) -> np.ndarray:
    if len(labels) <= 1:
        return np.array([], dtype=np.float64)
    return (labels[:-1] + labels[1:]) / 2.0


def scores_to_indices(scores: np.ndarray, labels: np.ndarray, step: float) -> np.ndarray:
    mn = float(labels[0])
    idx = np.rint((np.asarray(scores, dtype=np.float64) - mn) / step).astype(int)
    return np.clip(idx, 0, len(labels) - 1)


def apply_thresholds_to_raw_predictions(pred_raw: np.ndarray, thresholds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    pred_raw = np.asarray(pred_raw, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    class_idx = np.searchsorted(thresholds, pred_raw, side="right")
    class_idx = np.clip(class_idx, 0, len(labels) - 1)
    return labels[class_idx]


def qwk_for_thresholds(y_true_raw: np.ndarray, pred_raw: np.ndarray, thresholds: np.ndarray, labels: np.ndarray, step: float) -> float:
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


def make_candidate_values(pred_raw: np.ndarray, labels: np.ndarray, grid_size: int) -> np.ndarray:
    mn, mx = float(labels[0]), float(labels[-1])
    fixed_grid = np.linspace(mn, mx, max(grid_size, 3))
    default_th = default_midpoint_thresholds(labels)
    pred_raw = np.asarray(pred_raw, dtype=np.float64)
    pred_raw = pred_raw[~np.isnan(pred_raw)]
    pieces = [fixed_grid, default_th]
    if len(pred_raw) > 0:
        pieces.append(np.quantile(pred_raw, np.linspace(0.02, 0.98, 49)))
        uniq = np.unique(np.sort(pred_raw))
        if len(uniq) > 1:
            mids = (uniq[:-1] + uniq[1:]) / 2.0
            if len(mids) > 100:
                mids = np.quantile(mids, np.linspace(0.01, 0.99, 100))
            pieces.append(mids)
    candidates = np.concatenate(pieces)
    candidates = np.clip(candidates, mn, mx)
    return np.unique(np.round(candidates, 8))


def distribution_matching_init(y_true_raw: np.ndarray, pred_raw: np.ndarray, labels: np.ndarray, step: float) -> np.ndarray:
    if len(labels) <= 1:
        return np.array([], dtype=np.float64)
    y_idx = scores_to_indices(y_true_raw, labels, step)
    counts = np.bincount(y_idx, minlength=len(labels)).astype(np.float64)
    proportions = counts / max(counts.sum(), 1.0)
    cumulative = np.cumsum(proportions)[:-1]
    pred_raw = np.asarray(pred_raw, dtype=np.float64)
    if len(pred_raw) == 0 or np.all(np.isnan(pred_raw)):
        return default_midpoint_thresholds(labels)
    return np.sort(np.quantile(pred_raw, np.clip(cumulative, 0.0, 1.0)).astype(np.float64))


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
            "thresholds": [float(x) for x in default_th.tolist()],
            "labels": [float(x) for x in labels.tolist()],
            "dev_qwk_default": None if math.isnan(default_qwk) else float(default_qwk),
            "dev_qwk_tuned": None if math.isnan(default_qwk) else float(default_qwk),
            "n": int(len(y_true_raw)),
            "used_fallback_default": True,
            "reason": "Too few samples or only one gold class in k-specific dev split.",
        }

    candidates = make_candidate_values(pred_raw, labels, grid_size)
    rng = np.random.RandomState(seed)
    initializations = [
        default_th,
        distribution_matching_init(y_true_raw, pred_raw, labels, step),
    ]
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
        "reason": "optimized_on_k_specific_dev_qwk",
    }


def tune_thresholds_by_prompt_trait_from_predictions(
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
            gold_col = f"target_{trait}"
            pred_col = f"raw_pred_{trait}"
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
                seed=seed + abs(hash((prompt_id, trait))) % 100000,
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


def apply_threshold_map_to_rubric_prediction_df(
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
            pred_col = f"raw_pred_{trait}"
            tuned_col = f"pred_tuned_{trait}"
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


def compute_metrics_from_prediction_df(
    pred_df: pd.DataFrame,
    trait_cols: List[str],
    score_ranges: Dict[str, Dict[str, Dict[str, float]]],
    global_trait_fallback: Dict[str, Dict[str, float]],
    prompt_col: str,
    pred_prefix: str,
    round_step: float,
) -> Dict[str, Any]:
    trait_metrics: Dict[str, Any] = {}
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
            y_true_raw = subdf[gold_col].to_numpy(dtype=np.float64)
            y_pred_raw = subdf[pred_col].to_numpy(dtype=np.float64)
            y_true_idx = scores_to_indices(y_true_raw, labels, round_step)
            y_pred_idx = scores_to_indices(y_pred_raw, labels, round_step)
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
        trait_metrics[trait] = {"n": int(len(valid)), "qwk": qwk, "rmse": rmse}

    return {
        "mean_qwk": float(np.mean(qwk_values)) if qwk_values else float("nan"),
        "mean_rmse": float(np.mean(rmse_values)) if rmse_values else float("nan"),
        "trait_metrics": trait_metrics,
    }


def threshold_tune_rubric_predictions(
    dev_pred_df: pd.DataFrame,
    test_pred_df: pd.DataFrame,
    trait_cols: List[str],
    score_ranges: Dict[str, Dict[str, Dict[str, float]]],
    global_trait_fallback: Dict[str, Dict[str, float]],
    prompt_col: str,
    round_step: float,
    seed: int,
    grid_size: int,
    max_coord_iters: int,
    n_random_restarts: int,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    threshold_map = tune_thresholds_by_prompt_trait_from_predictions(
        dev_pred_df=dev_pred_df,
        trait_cols=trait_cols,
        score_ranges=score_ranges,
        global_trait_fallback=global_trait_fallback,
        prompt_col=prompt_col,
        round_step=round_step,
        seed=seed,
        grid_size=grid_size,
        max_coord_iters=max_coord_iters,
        n_random_restarts=n_random_restarts,
    )
    dev_tuned = apply_threshold_map_to_rubric_prediction_df(
        pred_df=dev_pred_df,
        threshold_map=threshold_map,
        trait_cols=trait_cols,
        score_ranges=score_ranges,
        global_trait_fallback=global_trait_fallback,
        prompt_col=prompt_col,
        round_step=round_step,
    )
    test_tuned = apply_threshold_map_to_rubric_prediction_df(
        pred_df=test_pred_df,
        threshold_map=threshold_map,
        trait_cols=trait_cols,
        score_ranges=score_ranges,
        global_trait_fallback=global_trait_fallback,
        prompt_col=prompt_col,
        round_step=round_step,
    )
    dev_default = compute_metrics_from_prediction_df(dev_tuned, trait_cols, score_ranges, global_trait_fallback, prompt_col, "pred", round_step)
    test_default = compute_metrics_from_prediction_df(test_tuned, trait_cols, score_ranges, global_trait_fallback, prompt_col, "pred", round_step)
    dev_thresholded = compute_metrics_from_prediction_df(dev_tuned, trait_cols, score_ranges, global_trait_fallback, prompt_col, "pred_tuned", round_step)
    test_thresholded = compute_metrics_from_prediction_df(test_tuned, trait_cols, score_ranges, global_trait_fallback, prompt_col, "pred_tuned", round_step)
    return threshold_map, dev_tuned, test_tuned, dev_default, test_default, dev_thresholded, test_thresholded


# ---------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------

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
            rows.append({
                "heldout_prompt": heldout_prompt,
                "fewshot_k": fewshot_k,
                "trait": trait,
                "mean_qwk": float(vals.mean()) if len(vals) else float("nan"),
                "std_qwk": float(vals.std(ddof=0)) if len(vals) else float("nan"),
                "num_repeats": int(len(vals)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

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
    mark_only_lora_cross_attention_trainable(model, freeze_shared_regressor=args.freeze_shared_regressor)
    param_counts = count_parameters(model)
    optimizer = build_optimizer(model, args)

    total_update_steps = max(1, math.ceil(len(train_loader) / args.grad_accum_steps) * args.num_epochs)
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_update_steps)

    amp_enabled, autocast_device, autocast_dtype = amp_context(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_dev_qwk = -1e9
    best_epoch = -1
    early_stop_counter = 0
    best_ckpt_path = os.path.join(run_dir, "best_rubric_fusion_traitwise_trainable.pt")
    history = []

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_reg_loss = 0.0
        running_contrastive_loss = 0.0
        num_steps = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}", leave=False)

        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            rubric_input_ids = batch["rubric_input_ids"].to(device)
            rubric_attention_mask = batch["rubric_attention_mask"].to(device)
            labels = batch["labels"].to(device)
            label_mask = batch["label_mask"].to(device)
            score_rubric_input_ids = batch.get("score_rubric_input_ids")
            score_rubric_attention_mask = batch.get("score_rubric_attention_mask")
            score_candidate_mask = batch.get("score_candidate_mask")
            score_candidate_values = batch.get("score_candidate_values")
            score_class_labels = batch.get("score_class_labels")
            if score_rubric_input_ids is not None:
                score_rubric_input_ids = score_rubric_input_ids.to(device)
                score_rubric_attention_mask = score_rubric_attention_mask.to(device)
                score_candidate_mask = score_candidate_mask.to(device)
                score_candidate_values = score_candidate_values.to(device)
                score_class_labels = score_class_labels.to(device)

            with torch.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=amp_enabled):
                model_out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    rubric_input_ids=rubric_input_ids,
                    rubric_attention_mask=rubric_attention_mask,
                    return_fused=args.use_rubric_score_contrastive,
                )
                if args.use_rubric_score_contrastive:
                    preds = model_out["preds"]
                    fused = model_out["fused"]
                else:
                    preds = model_out
                    fused = None

                reg_loss = masked_regression_loss(
                    preds=preds,
                    targets=labels,
                    mask=label_mask,
                    loss_type=args.loss_type,
                    huber_delta=args.huber_delta,
                )

                if args.use_rubric_score_contrastive:
                    if score_rubric_input_ids is None:
                        raise ValueError("score-level rubric tensors missing but --use_rubric_score_contrastive is enabled")
                    score_rubric_emb = model.encode_score_rubrics(
                        score_rubric_input_ids=score_rubric_input_ids,
                        score_rubric_attention_mask=score_rubric_attention_mask,
                    )
                    fused_for_contrastive, score_rubric_emb_for_contrastive = model.project_for_contrastive(
                        fused=fused,
                        score_rubric_emb=score_rubric_emb,
                    )
                    contrastive_loss = rubric_score_contrastive_loss(
                        fused=fused_for_contrastive,
                        score_rubric_emb=score_rubric_emb_for_contrastive,
                        score_class_labels=score_class_labels,
                        label_mask=label_mask,
                        score_candidate_mask=score_candidate_mask,
                        temperature=args.contrastive_temperature,
                        loss_type=args.contrastive_loss_type,
                        score_candidate_values=score_candidate_values,
                        soft_sigma=args.contrastive_soft_sigma,
                        hard_negative_weight=args.hard_negative_weight,
                        hard_negative_margin=args.hard_negative_margin,
                        hard_negative_top_k=args.hard_negative_top_k,
                        hard_negative_nearby_only=args.hard_negative_nearby_only,
                        hard_negative_nearby_distance=args.hard_negative_nearby_distance,
                    )
                    loss = reg_loss + args.contrastive_weight * contrastive_loss
                else:
                    contrastive_loss = torch.zeros((), device=device)
                    loss = reg_loss

                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running_loss += loss.item() * args.grad_accum_steps
            running_reg_loss += reg_loss.detach().item()
            running_contrastive_loss += contrastive_loss.detach().item()
            num_steps += 1
            progress.set_postfix(
                loss=f"{running_loss / max(num_steps, 1):.4f}",
                reg=f"{running_reg_loss / max(num_steps, 1):.4f}",
                con=f"{running_contrastive_loss / max(num_steps, 1):.4f}",
            )

        train_loss = running_loss / max(num_steps, 1)
        train_reg_loss = running_reg_loss / max(num_steps, 1)
        train_contrastive_loss = running_contrastive_loss / max(num_steps, 1)
        dev_metrics = evaluate_rubric_model(
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
            "train_reg_loss": train_reg_loss,
            "train_contrastive_loss": train_contrastive_loss,
            "dev_loss": dev_metrics["loss"],
            "dev_mean_qwk": dev_metrics["mean_qwk"],
            "dev_mean_rmse": dev_metrics["mean_rmse"],
            "rubric_fusion_scale": float(model.rubric_fusion_scale.detach().cpu()),
        })

        dev_qwk = dev_metrics["mean_qwk"]
        improved = not math.isnan(dev_qwk) and dev_qwk > best_dev_qwk
        if improved:
            best_dev_qwk = dev_qwk
            best_epoch = epoch
            early_stop_counter = 0
            torch.save({
                "trainable_state_dict": trainable_state_dict(model),
                "best_epoch": best_epoch,
                "best_dev_mean_qwk": best_dev_qwk,
                "trait_cols": trait_cols,
                "param_counts": param_counts,
                "architecture": model_architecture_name(model),
                "rubric_config": {
                    "rubric_model_name": args.rubric_model_name,
                    "rubric_max_length": args.rubric_max_length,
                    "rubric_text_field": args.rubric_text_field,
                    "include_source_excerpt": args.include_source_excerpt,
                    "source_excerpt_chars": args.source_excerpt_chars,
                    "include_prompt_id_in_essay_encoder": not args.exclude_prompt_id_in_essay_encoder,
                    "use_rubric_score_contrastive": args.use_rubric_score_contrastive,
                    "contrastive_weight": args.contrastive_weight,
                    "contrastive_temperature": args.contrastive_temperature,
                    "contrastive_loss_type": args.contrastive_loss_type,
                    "contrastive_soft_sigma": args.contrastive_soft_sigma,
                    "use_contrastive_projection": args.use_contrastive_projection,
                    "contrastive_projection_dropout": args.contrastive_projection_dropout,
                    "hard_negative_weight": args.hard_negative_weight,
                    "hard_negative_margin": args.hard_negative_margin,
                    "hard_negative_top_k": args.hard_negative_top_k,
                    "hard_negative_nearby_only": args.hard_negative_nearby_only,
                    "hard_negative_nearby_distance": args.hard_negative_nearby_distance,
                    "score_rubric_text_field": args.score_rubric_text_field,
                    "cross_attn_heads": args.cross_attn_heads,
                    "cross_attn_direction": model.cross_attn_direction,
                    "fusion_init": args.fusion_init,
                    "pooling_type": model.pooling_type,
                    "freeze_rubric_encoder": model.freeze_rubric_encoder,
                    "freeze_shared_regressor": args.freeze_shared_regressor,
                    "head_rubric_attention": model.head_rubric_attention,
                    "head_attn_heads": model.head_attn_heads,
                },
                "lora_config": {
                    "r": args.lora_r,
                    "alpha": args.lora_alpha,
                    "dropout": args.lora_dropout,
                    "target_modules": [x.strip() for x in args.lora_target_modules.split(",") if x.strip()],
                    "bias": args.lora_bias,
                    "use_rslora": args.use_rslora,
                    "use_dora": args.use_dora,
                },
            }, best_ckpt_path)
        else:
            early_stop_counter += 1

        if early_stop_counter >= args.patience:
            break

    if os.path.exists(best_ckpt_path):
        best_state = torch.load(best_ckpt_path, map_location=device)
        load_partial_state_dict(model, best_state["trainable_state_dict"], device=device)

    return model, history, best_epoch, best_dev_qwk, param_counts


# ---------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------

def checkpoint_is_rubric_fusion(ckpt: Dict[str, Any]) -> bool:
    state = ckpt.get("model_state_dict", {})
    keys = set(state.keys())
    return any(
        k.startswith("cross_attention.")
        or k.startswith("rubric_encoder.")
        or k.startswith("trait_embeddings.")
        or k.startswith("shared_regressor.")
        or k.startswith("trait_regressors.")
        or k in {"rubric_fusion_scale", "trait_fusion_scale"}
        for k in keys
    )



def model_architecture_name(model: nn.Module) -> str:
    head_kind = "traitwise_regression_heads" if getattr(model, "use_traitwise_regression_heads", False) else "shared_regressor"
    head_attn = "rubric_attentive_head" if getattr(model, "head_rubric_attention", False) else "no_head_rubric_attention"
    return f"rubric_lora_cross_attention_fusion_{head_kind}_{head_attn}"

def apply_lora_to_model_encoder(
    model: nn.Module,
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
    bias: str = "none",
    use_rslora: bool = False,
    use_dora: bool = False,
):
    """Apply LoRA directly to model.encoder.

    We do not use utils.apply_lora_to_encoder here because that helper was made
    for the old MultiTraitAESModel and uses modules_to_save=["regressor"].
    This rubric-aware architecture has `shared_regressor` or `trait_regressors`,
    not the old `regressor`, so LoRA should be applied only to the essay encoder.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError("PEFT is required. Install it with: pip install peft") from exc

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        use_rslora=use_rslora,
        use_dora=use_dora,
    )
    model.encoder = get_peft_model(model.encoder, lora_config)
    return model


def load_rubric_fusion_base_checkpoint(
    base_ckpt_dir: str,
    device: torch.device,
    args,
    trait_cols: List[str],
):
    """Load a rubric-aware base checkpoint produced by base_aes_rubric_fusion_traitwise.py."""
    ckpt_path = os.path.join(base_ckpt_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Rubric-aware base checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    loaded_trait_cols = ckpt.get("trait_cols", trait_cols)
    dropout = ckpt.get("dropout", 0.1)
    if args.dropout_override >= 0:
        dropout = args.dropout_override

    essay_model_name = ckpt.get("model_name", "roberta-base")
    rubric_config = ckpt.get("rubric_config", {})
    rubric_model_name = rubric_config.get("rubric_model_name", args.rubric_model_name)
    cross_attn_heads = rubric_config.get("cross_attn_heads", args.cross_attn_heads)
    fusion_init = rubric_config.get("fusion_init", ckpt.get("fusion_init", args.fusion_init))
    if args.pooling_type == "auto":
        pooling_type = rubric_config.get("pooling_type", ckpt.get("pooling_type", "mean"))
    else:
        pooling_type = args.pooling_type

    ckpt_cross_attn_direction = rubric_config.get(
        "cross_attn_direction",
        ckpt.get("cross_attn_direction", "essay_to_rubric"),
    )
    if args.cross_attn_direction == "auto":
        cross_attn_direction = ckpt_cross_attn_direction
    else:
        cross_attn_direction = args.cross_attn_direction
    if cross_attn_direction not in {"essay_to_rubric", "rubric_to_essay"}:
        raise ValueError(f"Unsupported cross_attn_direction loaded/resolved: {cross_attn_direction}")

    ckpt_head_attention = bool(rubric_config.get("head_rubric_attention", ckpt.get("head_rubric_attention", False)))
    head_rubric_attention = bool(args.head_rubric_attention or ckpt_head_attention)
    head_attn_heads = rubric_config.get("head_attn_heads", ckpt.get("head_attn_heads", args.head_attn_heads))
    if args.head_attn_heads > 0:
        head_attn_heads = args.head_attn_heads
    if head_attn_heads <= 0:
        head_attn_heads = cross_attn_heads

    # Build a small stub with an encoder so the adaptor model class can reuse it.
    base_stub = nn.Module()
    base_stub.encoder = AutoModel.from_pretrained(essay_model_name)

    state_keys = set(ckpt.get("model_state_dict", {}).keys())
    use_traitwise_regression_heads = any(k.startswith("trait_regressors.") for k in state_keys)
    if use_traitwise_regression_heads:
        print("Detected trait-wise regression heads in base checkpoint. Using trait_regressors in adaptor.")

    model = RubricFusionTraitWiseLoRAAESModel(
        base_model=base_stub,
        rubric_model_name=rubric_model_name,
        num_traits=len(loaded_trait_cols),
        dropout=dropout,
        cross_attn_heads=cross_attn_heads,
        cross_attn_direction=cross_attn_direction,
        fusion_init=fusion_init,
        freeze_rubric_encoder=not args.unfreeze_rubric_encoder,
        pooling_type=pooling_type,
        head_rubric_attention=head_rubric_attention,
        head_attn_heads=head_attn_heads,
        use_traitwise_regression_heads=use_traitwise_regression_heads,
        use_contrastive_projection=args.use_contrastive_projection,
        contrastive_projection_dropout=args.contrastive_projection_dropout,
    )

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # Before LoRA insertion, a matching rubric-aware checkpoint should load cleanly.
    # strict=False is used only to make the loader robust to minor version changes.
    important_missing = [
        k for k in missing
        if not k.startswith("rubric_projection.")
        and not k.startswith("contrastive_projection.")
    ]
    if important_missing:
        print(f"Warning: missing keys when loading rubric-aware base checkpoint: {important_missing[:20]}")
        if len(important_missing) > 20:
            print(f"... plus {len(important_missing) - 20} more missing keys")
    if unexpected:
        print(f"Warning: unexpected keys when loading rubric-aware base checkpoint: {unexpected[:20]}")
        if len(unexpected) > 20:
            print(f"... plus {len(unexpected) - 20} more unexpected keys")

    model.to(device)
    return model, loaded_trait_cols, ckpt


def build_rubric_aware_lora_model(base_ckpt_dir, device, args, trait_cols, lora_target_modules):
    """Build the adaptor model from either:
       1) the new rubric-aware base checkpoint, or
       2) the old non-rubric MultiTraitAESModel checkpoint.

    If the checkpoint is rubric-aware, we load it directly and then apply LoRA to
    model.encoder. This fixes the mismatch where a rubric-aware checkpoint was
    previously being loaded into MultiTraitAESModel.
    """
    ckpt_path = os.path.join(base_ckpt_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Base checkpoint not found: {ckpt_path}")
    ckpt_probe = torch.load(ckpt_path, map_location="cpu")

    if checkpoint_is_rubric_fusion(ckpt_probe):
        print("Detected rubric-aware base checkpoint. Loading RubricFusionTraitWise architecture.")
        model, loaded_trait_cols, _ = load_rubric_fusion_base_checkpoint(
            base_ckpt_dir=base_ckpt_dir,
            device=device,
            args=args,
            trait_cols=trait_cols,
        )
        trait_cols = loaded_trait_cols
    else:
        print("Detected old base checkpoint. Loading MultiTraitAESModel, then wrapping with rubric-fusion architecture.")
        base_model, loaded_trait_cols, _ = load_base_checkpoint_into_model(
            base_ckpt_dir=base_ckpt_dir,
            device=device,
            dropout_override=args.dropout_override,
        )
        trait_cols = loaded_trait_cols
        cross_attn_direction = "essay_to_rubric" if args.cross_attn_direction == "auto" else args.cross_attn_direction
        model = RubricFusionTraitWiseLoRAAESModel(
            base_model=base_model,
            rubric_model_name=args.rubric_model_name,
            num_traits=len(trait_cols),
            dropout=args.lora_dropout,
            cross_attn_heads=args.cross_attn_heads,
            cross_attn_direction=cross_attn_direction,
            fusion_init=args.fusion_init,
            freeze_rubric_encoder=not args.unfreeze_rubric_encoder,
            pooling_type="mean" if args.pooling_type == "auto" else args.pooling_type,
            head_rubric_attention=args.head_rubric_attention,
            head_attn_heads=args.head_attn_heads if args.head_attn_heads > 0 else args.cross_attn_heads,
            use_contrastive_projection=args.use_contrastive_projection,
            contrastive_projection_dropout=args.contrastive_projection_dropout,
        )
        model.to(device)

    model = apply_lora_to_model_encoder(
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
    return model, trait_cols


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    # Backward-compatible aliases from the residual-correction script.
    if getattr(args, "delta_init", None) is not None:
        args.fusion_init = args.delta_init
    if getattr(args, "freeze_base_regressor", False):
        args.freeze_shared_regressor = True
    ensure_dir(args.output_root)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Architecture: base encoder + LoRA + rubric encoder + trait-wise configurable cross-attention before shared regressor")
    print(f"Cross-attention direction request: {args.cross_attn_direction}")
    print(f"Pooling type request: {args.pooling_type}")
    print(f"Head rubric attention requested: {args.head_rubric_attention} (heads={args.head_attn_heads})")
    print(
        f"Rubric-score contrastive: {args.use_rubric_score_contrastive} "
        f"(type={args.contrastive_loss_type}, weight={args.contrastive_weight}, "
        f"temperature={args.contrastive_temperature}, soft_sigma={args.contrastive_soft_sigma}, "
        f"projection={args.use_contrastive_projection}, hard_negative_weight={args.hard_negative_weight})"
    )

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
    prompt_meta = load_prompt_meta(args.prompt_meta_json)
    score_ranges = build_score_ranges_from_hardcoded()
    global_trait_fallback = build_global_trait_fallback(full_df, TRAIT_COLUMNS)

    summary_rows = []
    repeat_test_rows = []

    for heldout_prompt in heldout_prompts:
        split_prompt_dir = os.path.join(args.split_root, f"heldout_{heldout_prompt}")
        base_ckpt_dir = os.path.join(args.base_root, f"{args.base_ckpt_prefix}{heldout_prompt}", "best_checkpoint")

        if not os.path.isdir(split_prompt_dir):
            print(f"Skipping heldout={heldout_prompt}: split dir not found -> {split_prompt_dir}")
            continue
        if not os.path.isdir(base_ckpt_dir):
            print(f"Skipping heldout={heldout_prompt}: base checkpoint dir not found -> {base_ckpt_dir}")
            continue

        repeat_dirs = sorted([
            os.path.join(split_prompt_dir, d)
            for d in os.listdir(split_prompt_dir)
            if d.startswith("repeat_") and os.path.isdir(os.path.join(split_prompt_dir, d))
        ])

        for repeat_dir in repeat_dirs:
            repeat_name = os.path.basename(repeat_dir)
            print(f"\n=== heldout={heldout_prompt} | {repeat_name} ===")

            test_path = os.path.join(repeat_dir, "test.tsv")
            if not os.path.exists(test_path):
                print(f"Skipping {repeat_name}: test split not found -> {test_path}")
                continue
            test_df = pd.read_csv(test_path, sep="\t")
            test_df[args.prompt_col] = test_df[args.prompt_col].apply(normalize_prompt_id)

            essay_tokenizer = AutoTokenizer.from_pretrained(base_ckpt_dir, use_fast=True)
            rubric_tokenizer = AutoTokenizer.from_pretrained(args.rubric_model_name, use_fast=True)

            # The test set stays fixed across k so different few-shot sizes remain comparable.
            test_dataset = RubricAwareAESDataset(
                df=test_df,
                essay_tokenizer=essay_tokenizer,
                rubric_tokenizer=rubric_tokenizer,
                trait_cols=TRAIT_COLUMNS,
                prompt_col=args.prompt_col,
                text_col=args.text_col,
                prompt_text_map=prompt_text_map,
                score_ranges=score_ranges,
                global_trait_fallback=global_trait_fallback,
                max_length=args.max_length,
                rubric_max_length=args.rubric_max_length,
                prompt_meta=prompt_meta,
                rubric_text_field=args.rubric_text_field,
                include_source_excerpt=args.include_source_excerpt,
                source_excerpt_chars=args.source_excerpt_chars,
                include_prompt_id_in_essay_encoder=not args.exclude_prompt_id_in_essay_encoder,
                score_rubric_text_field=args.score_rubric_text_field,
            )
            test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

            repeat_out_dir = os.path.join(args.output_root, f"heldout_{heldout_prompt}", repeat_name)
            ensure_dir(repeat_out_dir)

            for k in fewshot_sizes:
                train_path = os.path.join(repeat_dir, f"fewshot_{k}.tsv")
                dev_path = os.path.join(repeat_dir, args.dev_file_template.format(k=k))

                if not os.path.exists(train_path):
                    print(f"Skipping k={k}: train split not found -> {train_path}")
                    continue
                if not os.path.exists(dev_path):
                    print(f"Skipping k={k}: k-specific dev split not found -> {dev_path}")
                    continue

                train_df = pd.read_csv(train_path, sep="\t")
                dev_df = pd.read_csv(dev_path, sep="\t")
                train_df[args.prompt_col] = train_df[args.prompt_col].apply(normalize_prompt_id)
                dev_df[args.prompt_col] = dev_df[args.prompt_col].apply(normalize_prompt_id)

                if len(train_df) == 0:
                    print(f"Skipping k={k}: empty train split")
                    continue
                if len(dev_df) == 0:
                    print(f"Skipping k={k}: empty k-specific dev split")
                    continue

                dev_dataset = RubricAwareAESDataset(
                    df=dev_df,
                    essay_tokenizer=essay_tokenizer,
                    rubric_tokenizer=rubric_tokenizer,
                    trait_cols=TRAIT_COLUMNS,
                    prompt_col=args.prompt_col,
                    text_col=args.text_col,
                    prompt_text_map=prompt_text_map,
                    score_ranges=score_ranges,
                    global_trait_fallback=global_trait_fallback,
                    max_length=args.max_length,
                    rubric_max_length=args.rubric_max_length,
                    prompt_meta=prompt_meta,
                    rubric_text_field=args.rubric_text_field,
                    include_source_excerpt=args.include_source_excerpt,
                    source_excerpt_chars=args.source_excerpt_chars,
                    include_prompt_id_in_essay_encoder=not args.exclude_prompt_id_in_essay_encoder,
                    score_rubric_text_field=args.score_rubric_text_field,
                )
                dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

                train_dataset = RubricAwareAESDataset(
                    df=train_df,
                    essay_tokenizer=essay_tokenizer,
                    rubric_tokenizer=rubric_tokenizer,
                    trait_cols=TRAIT_COLUMNS,
                    prompt_col=args.prompt_col,
                    text_col=args.text_col,
                    prompt_text_map=prompt_text_map,
                    score_ranges=score_ranges,
                    global_trait_fallback=global_trait_fallback,
                    max_length=args.max_length,
                    rubric_max_length=args.rubric_max_length,
                    prompt_meta=prompt_meta,
                    rubric_text_field=args.rubric_text_field,
                    include_source_excerpt=args.include_source_excerpt,
                    source_excerpt_chars=args.source_excerpt_chars,
                    include_prompt_id_in_essay_encoder=not args.exclude_prompt_id_in_essay_encoder,
                    score_rubric_text_field=args.score_rubric_text_field,
                )
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

                # Trait cols are read from the base checkpoint, so build the model before training.
                # Use TRAIT_COLUMNS initially because this is how the dataset is constructed; the checkpoint will confirm actual columns.
                model, trait_cols = build_rubric_aware_lora_model(
                    base_ckpt_dir=base_ckpt_dir,
                    device=device,
                    args=args,
                    trait_cols=TRAIT_COLUMNS,
                    lora_target_modules=lora_target_modules,
                )

                # If base checkpoint trait_cols are different, rebuild datasets with that order.
                if list(trait_cols) != list(TRAIT_COLUMNS):
                    print(f"Rebuilding datasets with checkpoint trait_cols: {trait_cols}")
                    train_dataset = RubricAwareAESDataset(
                        df=train_df, essay_tokenizer=essay_tokenizer, rubric_tokenizer=rubric_tokenizer,
                        trait_cols=trait_cols, prompt_col=args.prompt_col, text_col=args.text_col,
                        prompt_text_map=prompt_text_map, score_ranges=score_ranges,
                        global_trait_fallback=global_trait_fallback, max_length=args.max_length,
                        rubric_max_length=args.rubric_max_length, prompt_meta=prompt_meta,
                        rubric_text_field=args.rubric_text_field,
                        include_source_excerpt=args.include_source_excerpt,
                        source_excerpt_chars=args.source_excerpt_chars,
                        include_prompt_id_in_essay_encoder=not args.exclude_prompt_id_in_essay_encoder,
                        score_rubric_text_field=args.score_rubric_text_field,
                    )
                    dev_dataset = RubricAwareAESDataset(
                        df=dev_df, essay_tokenizer=essay_tokenizer, rubric_tokenizer=rubric_tokenizer,
                        trait_cols=trait_cols, prompt_col=args.prompt_col, text_col=args.text_col,
                        prompt_text_map=prompt_text_map, score_ranges=score_ranges,
                        global_trait_fallback=global_trait_fallback, max_length=args.max_length,
                        rubric_max_length=args.rubric_max_length, prompt_meta=prompt_meta,
                        rubric_text_field=args.rubric_text_field,
                        include_source_excerpt=args.include_source_excerpt,
                        source_excerpt_chars=args.source_excerpt_chars,
                        include_prompt_id_in_essay_encoder=not args.exclude_prompt_id_in_essay_encoder,
                        score_rubric_text_field=args.score_rubric_text_field,
                    )
                    test_dataset = RubricAwareAESDataset(
                        df=test_df, essay_tokenizer=essay_tokenizer, rubric_tokenizer=rubric_tokenizer,
                        trait_cols=trait_cols, prompt_col=args.prompt_col, text_col=args.text_col,
                        prompt_text_map=prompt_text_map, score_ranges=score_ranges,
                        global_trait_fallback=global_trait_fallback, max_length=args.max_length,
                        rubric_max_length=args.rubric_max_length, prompt_meta=prompt_meta,
                        rubric_text_field=args.rubric_text_field,
                        include_source_excerpt=args.include_source_excerpt,
                        source_excerpt_chars=args.source_excerpt_chars,
                        include_prompt_id_in_essay_encoder=not args.exclude_prompt_id_in_essay_encoder,
                        score_rubric_text_field=args.score_rubric_text_field,
                    )
                    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
                    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
                    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

                run_dir = os.path.join(repeat_out_dir, f"k_{k}")
                ensure_dir(run_dir)

                # Zero-shot evaluation on the exact same k-specific dev split
                # and fixed repeat test split, before any few-shot adaptation.
                # This does not change the model architecture or training logic;
                # it only records the pre-adaptation baseline for fair comparison.
                zero_shot_dev = evaluate_rubric_model(
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
                zero_shot_test = evaluate_rubric_model(
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
                zero_shot_dev_predictions = predict_to_dataframe_rubric(
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
                zero_shot_test_predictions = predict_to_dataframe_rubric(
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
                save_json(zero_shot_dev, os.path.join(run_dir, "zero_shot_dev_metrics.json"))
                save_json(zero_shot_test, os.path.join(run_dir, "zero_shot_test_metrics.json"))
                zero_shot_dev_predictions.to_csv(os.path.join(run_dir, "zero_shot_dev_predictions.csv"), index=False)
                zero_shot_test_predictions.to_csv(os.path.join(run_dir, "zero_shot_test_predictions.csv"), index=False)

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

                final_dev = evaluate_rubric_model(
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
                final_test = evaluate_rubric_model(
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

                final_dev_predictions = predict_to_dataframe_rubric(
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
                final_test_predictions = predict_to_dataframe_rubric(
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

                threshold_outputs: Dict[str, Any] = {}
                if not args.disable_threshold_tuning:
                    (
                        threshold_map,
                        final_dev_predictions_thresholded,
                        final_test_predictions_thresholded,
                        threshold_dev_default,
                        threshold_test_default,
                        threshold_dev_tuned,
                        threshold_test_tuned,
                    ) = threshold_tune_rubric_predictions(
                        dev_pred_df=final_dev_predictions,
                        test_pred_df=final_test_predictions,
                        trait_cols=trait_cols,
                        score_ranges=score_ranges,
                        global_trait_fallback=global_trait_fallback,
                        prompt_col=args.prompt_col,
                        round_step=args.round_step,
                        seed=args.seed + int(k),
                        grid_size=args.threshold_grid_size,
                        max_coord_iters=args.threshold_max_coord_iters,
                        n_random_restarts=args.threshold_random_restarts,
                    )
                    save_json(threshold_map, os.path.join(run_dir, "thresholds_from_dev_k.json"))
                    save_json(threshold_dev_default, os.path.join(run_dir, "threshold_dev_default_rounding_metrics.json"))
                    save_json(threshold_test_default, os.path.join(run_dir, "threshold_test_default_rounding_metrics.json"))
                    save_json(threshold_dev_tuned, os.path.join(run_dir, "threshold_dev_tuned_metrics.json"))
                    save_json(threshold_test_tuned, os.path.join(run_dir, "threshold_test_tuned_metrics.json"))
                    final_dev_predictions_thresholded.to_csv(os.path.join(run_dir, "final_dev_predictions_thresholded.csv"), index=False)
                    final_test_predictions_thresholded.to_csv(os.path.join(run_dir, "final_test_predictions_thresholded.csv"), index=False)
                    threshold_outputs = {
                        "threshold_dev_default": threshold_dev_default,
                        "threshold_test_default": threshold_test_default,
                        "threshold_dev_tuned": threshold_dev_tuned,
                        "threshold_test_tuned": threshold_test_tuned,
                    }

                save_json({"history": history}, os.path.join(run_dir, "training_history.json"))
                save_json(final_dev, os.path.join(run_dir, "final_dev_metrics.json"))
                save_json(final_test, os.path.join(run_dir, "final_test_metrics.json"))
                final_dev_predictions.to_csv(os.path.join(run_dir, "final_dev_predictions.csv"), index=False)
                final_test_predictions.to_csv(os.path.join(run_dir, "final_test_predictions.csv"), index=False)
                save_json({
                    "heldout_prompt": heldout_prompt,
                    "repeat_name": repeat_name,
                    "fewshot_k": k,
                    "best_epoch": best_epoch,
                    "best_dev_mean_qwk": best_dev_qwk,
                    "train_path": train_path,
                    "dev_path": dev_path,
                    "test_path": test_path,
                    "dev_file_template": args.dev_file_template,
                    "base_checkpoint_dir": base_ckpt_dir,
                    "prompt_meta_json": args.prompt_meta_json,
                    "architecture": model_architecture_name(model),
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "lora_target_modules": lora_target_modules,
                    "lora_bias": args.lora_bias,
                    "rubric_model_name": args.rubric_model_name,
                    "rubric_max_length": args.rubric_max_length,
                    "rubric_text_field": args.rubric_text_field,
                    "include_source_excerpt": args.include_source_excerpt,
                    "source_excerpt_chars": args.source_excerpt_chars,
                    "include_prompt_id_in_essay_encoder": not args.exclude_prompt_id_in_essay_encoder,
                    "use_rubric_score_contrastive": args.use_rubric_score_contrastive,
                    "contrastive_weight": args.contrastive_weight,
                    "contrastive_temperature": args.contrastive_temperature,
                    "contrastive_loss_type": args.contrastive_loss_type,
                    "contrastive_soft_sigma": args.contrastive_soft_sigma,
                    "use_contrastive_projection": args.use_contrastive_projection,
                    "contrastive_projection_dropout": args.contrastive_projection_dropout,
                    "hard_negative_weight": args.hard_negative_weight,
                    "hard_negative_margin": args.hard_negative_margin,
                    "hard_negative_top_k": args.hard_negative_top_k,
                    "hard_negative_nearby_only": args.hard_negative_nearby_only,
                    "hard_negative_nearby_distance": args.hard_negative_nearby_distance,
                    "score_rubric_text_field": args.score_rubric_text_field,
                    "cross_attn_heads": args.cross_attn_heads,
                    "cross_attn_direction": model.cross_attn_direction,
                    "fusion_init": args.fusion_init,
                    "pooling_type": model.pooling_type,
                    "head_rubric_attention": model.head_rubric_attention,
                    "head_attn_heads": model.head_attn_heads,
                    "param_counts": param_counts,
                    "train_n": len(train_df),
                    "dev_n": len(dev_df),
                    "test_n": len(test_df),
                    "use_dora": args.use_dora,
                    "use_rslora": args.use_rslora,
                    "freeze_shared_regressor": args.freeze_shared_regressor,
                    "unfreeze_rubric_encoder": args.unfreeze_rubric_encoder,
                    "zero_shot_dev_mean_qwk": zero_shot_dev["mean_qwk"],
                    "zero_shot_dev_mean_rmse": zero_shot_dev["mean_rmse"],
                    "zero_shot_test_mean_qwk": zero_shot_test["mean_qwk"],
                    "zero_shot_test_mean_rmse": zero_shot_test["mean_rmse"],
                    "adapted_minus_zero_shot_dev_qwk": final_dev["mean_qwk"] - zero_shot_dev["mean_qwk"],
                    "adapted_minus_zero_shot_test_qwk": final_test["mean_qwk"] - zero_shot_test["mean_qwk"],
                    "threshold_tuning_enabled": not args.disable_threshold_tuning,
                    "threshold_grid_size": args.threshold_grid_size,
                    "threshold_max_coord_iters": args.threshold_max_coord_iters,
                    "threshold_random_restarts": args.threshold_random_restarts,
                }, os.path.join(run_dir, "run_config.json"))

                print(f"\nRubric-LoRA-FusionTraitWise | heldout={heldout_prompt} | {repeat_name} | k={k}")
                print(f"Trainable params: {param_counts['trainable']:,} / {param_counts['total']:,}")
                print(format_metrics_for_print("Zero-shot dev", zero_shot_dev))
                print(format_metrics_for_print("Zero-shot test", zero_shot_test))
                print(format_metrics_for_print("Final dev", final_dev))
                print(format_metrics_for_print("Final test", final_test))
                if threshold_outputs:
                    print(format_metrics_for_print("Threshold-tuned dev", threshold_outputs["threshold_dev_tuned"]))
                    print(format_metrics_for_print("Threshold-tuned test", threshold_outputs["threshold_test_tuned"]))

                base_row = {
                    "heldout_prompt": heldout_prompt,
                    "repeat_name": repeat_name,
                    "fewshot_k": k,
                    "train_n": len(train_df),
                    "dev_n": len(dev_df),
                    "test_n": len(test_df),
                    "best_epoch": best_epoch,
                    "best_dev_mean_qwk": best_dev_qwk,
                    "zero_shot_dev_mean_qwk": zero_shot_dev["mean_qwk"],
                    "zero_shot_dev_mean_rmse": zero_shot_dev["mean_rmse"],
                    "zero_shot_test_mean_qwk": zero_shot_test["mean_qwk"],
                    "zero_shot_test_mean_rmse": zero_shot_test["mean_rmse"],
                    "final_dev_mean_qwk": final_dev["mean_qwk"],
                    "final_dev_mean_rmse": final_dev["mean_rmse"],
                    "final_test_mean_qwk": final_test["mean_qwk"],
                    "final_test_mean_rmse": final_test["mean_rmse"],
                    "adapted_minus_zero_shot_dev_qwk": final_dev["mean_qwk"] - zero_shot_dev["mean_qwk"],
                    "adapted_minus_zero_shot_test_qwk": final_test["mean_qwk"] - zero_shot_test["mean_qwk"],
                    "threshold_tuning_enabled": not args.disable_threshold_tuning,
                    "final_dev_tuned_mean_qwk": threshold_outputs.get("threshold_dev_tuned", {}).get("mean_qwk", float("nan")),
                    "final_dev_tuned_mean_rmse": threshold_outputs.get("threshold_dev_tuned", {}).get("mean_rmse", float("nan")),
                    "final_test_tuned_mean_qwk": threshold_outputs.get("threshold_test_tuned", {}).get("mean_qwk", float("nan")),
                    "final_test_tuned_mean_rmse": threshold_outputs.get("threshold_test_tuned", {}).get("mean_rmse", float("nan")),
                    "trainable_params": param_counts["trainable"],
                    "total_params": param_counts["total"],
                    "rubric_fusion_scale_final": float(model.rubric_fusion_scale.detach().cpu()),
                    "cross_attn_direction": model.cross_attn_direction,
                    "head_rubric_attention": model.head_rubric_attention,
                }
                base_row.update(flatten_trait_metrics("zero_shot_dev", zero_shot_dev["trait_metrics"], trait_cols))
                base_row.update(flatten_trait_metrics("zero_shot_test", zero_shot_test["trait_metrics"], trait_cols))
                base_row.update(flatten_trait_metrics("dev", final_dev["trait_metrics"], trait_cols))
                base_row.update(flatten_trait_metrics("test", final_test["trait_metrics"], trait_cols))
                if threshold_outputs:
                    base_row.update(flatten_trait_metrics("dev_tuned", threshold_outputs["threshold_dev_tuned"]["trait_metrics"], trait_cols))
                    base_row.update(flatten_trait_metrics("test_tuned", threshold_outputs["threshold_test_tuned"]["trait_metrics"], trait_cols))
                summary_rows.append(base_row)
                repeat_test_rows.append(base_row.copy())

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(os.path.join(args.output_root, "rubric_lora_fusion_traitwise_summary.csv"), index=False)

    if repeat_test_rows:
        repeat_test_df = pd.DataFrame(repeat_test_rows)
        repeat_test_df.to_csv(os.path.join(args.output_root, "rubric_lora_fusion_traitwise_repeat_test_results.csv"), index=False)
        mean_trait_qwk_df = build_mean_trait_qwk_across_repeats(
            repeat_test_df,
            trait_cols=TRAIT_COLUMNS,
            score_column_prefix="test",
        )
        mean_trait_qwk_df.to_csv(os.path.join(args.output_root, "rubric_lora_fusion_traitwise_mean_trait_qwk_across_repeats.csv"), index=False)

        zero_shot_mean_trait_qwk_df = build_mean_trait_qwk_across_repeats(
            repeat_test_df,
            trait_cols=TRAIT_COLUMNS,
            score_column_prefix="zero_shot_test",
        )
        zero_shot_mean_trait_qwk_df.to_csv(
            os.path.join(args.output_root, "rubric_lora_fusion_traitwise_zero_shot_mean_trait_qwk_across_repeats.csv"),
            index=False,
        )

        if any(c.startswith("test_tuned_") and c.endswith("_qwk") for c in repeat_test_df.columns):
            tuned_mean_trait_qwk_df = build_mean_trait_qwk_across_repeats(
                repeat_test_df,
                trait_cols=TRAIT_COLUMNS,
                score_column_prefix="test_tuned",
            )
            tuned_mean_trait_qwk_df.to_csv(
                os.path.join(args.output_root, "rubric_lora_fusion_traitwise_threshold_tuned_mean_trait_qwk_across_repeats.csv"),
                index=False,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
