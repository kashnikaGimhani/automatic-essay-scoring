#!/usr/bin/env python3
"""
Rubric-aware base training for prompt-shift AES.

Architecture: Option A — shared scalar regressor applied trait-by-trait.

Compared with the original base_aes.py:
  - The essay encoder is still the initial base encoder from utils.MODEL_NAME (roberta-base).
  - A separate rubric encoder reads prompt-trait rubric text from asap_prompt_meta_v3.json.
  - Main cross-attention can run essay→rubric or rubric→essay using --cross_attn_direction.
  - For each trait, the model builds a fused representation BEFORE scoring:
        essay_pooled + rubric_guidance + trait_embedding
  - A rubric-attentive shared scalar regressor can re-attend to the same prompt-trait rubric tokens before producing each trait score.

This pre-trains the cross-attention branch on source prompts, so later LoRA adaptation can
start from a rubric-aware checkpoint instead of learning cross-attention from few-shot data only.
"""

import os
import math
import json
import argparse
import shutil
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import cohen_kappa_score, mean_squared_error

import sys
from pathlib import Path

PARENT_DIR = str(Path(__file__).resolve().parents[1])
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from utils import (
    MODEL_NAME,
    TRAIT_COLUMNS,
    set_seed,
    ensure_dir,
    save_json,
    load_json,
    normalize_prompt_id,
    build_prompt_text_map,
    build_score_ranges_from_hardcoded,
    build_global_trait_fallback,
    split_source_by_prompt,
    masked_regression_loss,
    format_metrics_for_print,
    get_range_for_trait,
    normalize_score,
    denormalize_score,
    round_to_step,
    masked_mean_pool,
)

print("Libraries imported successfully.", flush=True)


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rubric-aware base training for prompt-shift AES with trait-wise cross-attention fusion"
    )

    parser.add_argument("--data_path", type=str, required=True, help="Path to TSV/CSV data file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--prompt_meta_json", type=str, required=True, help="Path to asap_prompt_meta_v3.json")
    parser.add_argument("--sep", type=str, default="\t", help="File separator, default is tab")

    parser.add_argument("--heldout_prompt", type=str, required=True, help="Prompt ID to hold out")
    parser.add_argument("--prompt_col", type=str, default="essay_set")
    parser.add_argument("--text_col", type=str, default="essay")
    parser.add_argument("--id_col", type=str, default="essay_id")

    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_length", type=int, default=480, help="Essay encoder max length")
    parser.add_argument("--rubric_max_length", type=int, default=256, help="Rubric encoder max length")
    parser.add_argument("--rubric_model_name", type=str, default="roberta-base")
    parser.add_argument("--rubric_text_field", type=str, default="trait_rubric_encoder_text")
    parser.add_argument("--include_source_excerpt", action="store_true", help="Append short source excerpt to rubric text for source-dependent prompts")
    parser.add_argument("--source_excerpt_chars", type=int, default=1200)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10)

    # LR defaults: encoder uses small LR; newly initialized cross-attention/regressor use larger LR.
    parser.add_argument("--lr", type=float, default=2e-5, help="Essay encoder learning rate")
    parser.add_argument("--cross_attn_lr", type=float, default=1e-4, help="Cross-attention/fusion learning rate")
    parser.add_argument("--regressor_lr", type=float, default=1e-4, help="Shared trait regressor learning rate")
    parser.add_argument("--rubric_lr", type=float, default=1e-5, help="Rubric encoder LR if --unfreeze_rubric_encoder is used")

    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=3)

    parser.add_argument("--cross_attn_heads", type=int, default=8)
    parser.add_argument("--fusion_init", type=float, default=0.1, help="Initial scale for rubric and trait fusion")
    parser.add_argument(
        "--pooling_type",
        type=str,
        default="mean",
        choices=["mean", "attention"],
        help="Pooling for essay states and cross-attention outputs. Use 'attention' for learned attention pooling.",
    )
    parser.add_argument(
        "--cross_attn_direction",
        type=str,
        default="essay_to_rubric",
        choices=["essay_to_rubric", "rubric_to_essay"],
        help=(
            "Cross-attention direction. "
            "essay_to_rubric: essay tokens query rubric tokens. "
            "rubric_to_essay: rubric tokens query essay tokens."
        ),
    )
    parser.add_argument(
        "--head_rubric_attention",
        action="store_true",
        help="Make the shared scoring head attend to rubric tokens before predicting each trait score.",
    )
    parser.add_argument(
        "--head_attn_heads",
        type=int,
        default=-1,
        help="Number of heads for rubric attention inside the scoring head. Use -1 to reuse --cross_attn_heads.",
    )
    parser.add_argument("--unfreeze_rubric_encoder", action="store_true", help="Optional ablation: train rubric encoder too")
    parser.add_argument("--freeze_essay_encoder", action="store_true", help="Optional ablation: train only fusion/regressor, not essay encoder")

    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "huber"])
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument(
        "--round_step",
        type=float,
        default=1.0,
        help="Step for rounding scores during QWK eval, e.g. 1.0 or 0.5",
    )

    parser.add_argument("--save_split_files", action="store_true")
    parser.add_argument("--run_zero_shot_eval", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------
# Prompt metadata / rubric text helpers
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
        text = pmeta[rubric_text_field].get(trait, "")

    # Fallback 1: score-level texts.
    if not text and isinstance(pmeta.get("cross_attention_score_texts"), dict):
        score_texts = pmeta["cross_attention_score_texts"].get(trait, {})
        if isinstance(score_texts, dict) and score_texts:
            ordered_items = sorted(score_texts.items(), key=lambda kv: float(kv[0]))
            joined = " ".join([f"Score {score}: {direction}" for score, direction in ordered_items])
            text = f"Prompt {prompt_id}. Trait: {trait_text}. Score-level directions: {joined}"

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
    ):
        self.df = df.reset_index(drop=True).copy()
        self.trait_cols = trait_cols
        self.prompt_col = prompt_col
        self.text_col = text_col
        self.score_ranges = score_ranges
        self.global_trait_fallback = global_trait_fallback
        self.prompt_meta = prompt_meta

        self.prompt_ids = [normalize_prompt_id(x) for x in self.df[prompt_col].tolist()]
        prompt_texts = [prompt_text_map.get(pid, f"Prompt {pid}") for pid in self.prompt_ids]
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

        # Labels and masks.
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

        # Cache rubric tensors per unique prompt. This avoids storing [N,T,L] copies.
        self.prompt_rubric_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        unique_prompt_ids = sorted(set(self.prompt_ids))
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pid = self.prompt_ids[idx]
        cached = self.prompt_rubric_cache[pid]
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "rubric_input_ids": cached["input_ids"],
            "rubric_attention_mask": cached["attention_mask"],
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

    Unlike mean pooling, this layer learns which tokens should receive more
    weight when forming the essay/rubric-guided representation.
    """

    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.scorer(hidden_states).squeeze(-1)  # [B, L]
        mask = attention_mask.bool()
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = weights.masked_fill(~mask, 0.0)
        denom = weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        weights = weights / denom
        return torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)


class SharedTraitRegressor(nn.Module):
    """One shared scalar scorer applied to every trait-specific fused representation.

    When use_rubric_attention=True, the scoring head itself attends back to the
    prompt-trait rubric tokens before predicting the scalar score. This means the
    rubric is used twice:
      1. essay-token cross-attention creates rubric-guided context
      2. head-level attention lets the final scorer re-check rubric criteria
    """

    def __init__(
        self,
        hidden: int,
        dropout: float,
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

    def forward(
        self,
        x: torch.Tensor,
        rubric_hidden: Optional[torch.Tensor] = None,
        rubric_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_rubric_attention:
            if rubric_hidden is None or rubric_mask is None:
                raise ValueError("rubric_hidden and rubric_mask are required when use_rubric_attention=True")

            # x: [B*T, H] becomes a one-token query for the scoring head.
            query = x.unsqueeze(1)  # [B*T, 1, H]
            attn_out, _ = self.head_rubric_attention(
                query=query,
                key=rubric_hidden,
                value=rubric_hidden,
                key_padding_mask=~rubric_mask.bool(),
                need_weights=False,
            )
            attn_out = attn_out.squeeze(1)  # [B*T, H]

            # Residual update from rubric tokens.
            x = self.head_attn_norm(x + self.head_dropout(attn_out))

            # Small head-level FFN refinement.
            ffn_out = self.head_ffn(x)
            x = self.head_ffn_norm(x + self.head_dropout(ffn_out))

        return self.net(x)


class RubricFusionTraitWiseAESModel(nn.Module):
    """Rubric-aware base AES model with cross-attention fusion before regression."""

    def __init__(
        self,
        essay_model_name: str,
        rubric_model_name: str,
        num_traits: int,
        dropout: float = 0.1,
        cross_attn_heads: int = 8,
        fusion_init: float = 0.1,
        freeze_rubric_encoder: bool = True,
        pooling_type: str = "mean",
        cross_attn_direction: str = "essay_to_rubric",
        head_rubric_attention: bool = False,
        head_attn_heads: int = 8,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(essay_model_name)
        self.rubric_encoder = AutoModel.from_pretrained(rubric_model_name)
        self.num_traits = num_traits
        self.freeze_rubric_encoder = freeze_rubric_encoder

        hidden = self.encoder.config.hidden_size
        rubric_hidden = self.rubric_encoder.config.hidden_size
        self.hidden_size = hidden
        self.pooling_type = pooling_type
        self.cross_attn_direction = cross_attn_direction
        if self.cross_attn_direction not in {"essay_to_rubric", "rubric_to_essay"}:
            raise ValueError(f"Unsupported cross_attn_direction: {self.cross_attn_direction}")
        self.head_rubric_attention = head_rubric_attention
        self.head_attn_heads = head_attn_heads

        if pooling_type == "attention":
            self.essay_pooler = MaskedAttentionPooling(hidden, dropout=dropout)
            self.context_pooler = MaskedAttentionPooling(hidden, dropout=dropout)
        elif pooling_type == "mean":
            self.essay_pooler = None
            self.context_pooler = None
        else:
            raise ValueError(f"Unsupported pooling_type: {pooling_type}")

        if hidden % cross_attn_heads != 0:
            raise ValueError(f"hidden_size={hidden} must be divisible by cross_attn_heads={cross_attn_heads}")

        self.rubric_projection = nn.Identity() if rubric_hidden == hidden else nn.Linear(rubric_hidden, hidden)
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
        self.shared_regressor = SharedTraitRegressor(
            hidden=hidden,
            dropout=dropout,
            use_rubric_attention=head_rubric_attention,
            head_attn_heads=head_attn_heads,
        )

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
        # rubric_input_ids: [B, T, R]
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

        return self.rubric_projection(hidden)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rubric_input_ids: Optional[torch.Tensor] = None,
        rubric_attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        essay_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        essay_hidden = essay_out.last_hidden_state                 # [B, S, H]
        essay_pooled = self.pool_tokens(essay_hidden, attention_mask, which="essay")  # [B, H]
        bsz, seq_len, hidden = essay_hidden.shape

        # Fallback path if rubric tensors are not supplied.
        if rubric_input_ids is None or rubric_attention_mask is None:
            trait_ids = torch.arange(self.num_traits, device=input_ids.device).unsqueeze(0).expand(bsz, self.num_traits)
            trait_vec = self.trait_embeddings(trait_ids)  # [B, T, H]
            essay_rep = essay_pooled.unsqueeze(1).expand(bsz, self.num_traits, hidden)
            fused = self.fusion_norm(essay_rep + self.trait_fusion_scale * trait_vec)
            fused_flat = fused.reshape(bsz * self.num_traits, hidden)
            if self.head_rubric_attention:
                raise ValueError("rubric tensors are required because head_rubric_attention=True")
            return self.shared_regressor(fused_flat).reshape(bsz, self.num_traits)

        num_traits = rubric_input_ids.shape[1]
        if num_traits != self.num_traits:
            raise ValueError(f"rubric num_traits={num_traits}, expected={self.num_traits}")

        rubric_hidden = self.encode_rubric(rubric_input_ids, rubric_attention_mask)  # [B*T, R, H]
        rubric_mask = rubric_attention_mask.reshape(bsz * num_traits, rubric_attention_mask.shape[-1]).bool()

        # Repeat essay states for each trait.
        essay_rep = essay_hidden.unsqueeze(1).expand(bsz, num_traits, seq_len, hidden).reshape(bsz * num_traits, seq_len, hidden)
        essay_mask_rep = attention_mask.unsqueeze(1).expand(bsz, num_traits, seq_len).reshape(bsz * num_traits, seq_len)
        essay_pooled_rep = essay_pooled.unsqueeze(1).expand(bsz, num_traits, hidden).reshape(bsz * num_traits, hidden)

        trait_ids = torch.arange(num_traits, device=input_ids.device).unsqueeze(0).expand(bsz, num_traits).reshape(-1)
        trait_vec = self.trait_embeddings(trait_ids)  # [B*T, H]

        # Main cross-attention fusion block.
        # The output length follows the query sequence, so the pooling mask changes by direction.
        if self.cross_attn_direction == "essay_to_rubric":
            # Essay-centered direction:
            #   Q = essay tokens, K/V = rubric tokens
            # Meaning: which rubric criteria are relevant to this essay?
            query = essay_rep + trait_vec.unsqueeze(1)
            attn_out, _ = self.cross_attention(
                query=query,
                key=rubric_hidden,
                value=rubric_hidden,
                key_padding_mask=~rubric_mask,
                need_weights=False,
            )
            # attn_out shape: [B*T, essay_len, H], so pool using essay_mask_rep.
            rubric_context = self.pool_tokens(attn_out, essay_mask_rep, which="context")  # [B*T, H]

        elif self.cross_attn_direction == "rubric_to_essay":
            # Rubric-centered direction:
            #   Q = rubric tokens, K/V = essay tokens
            # Meaning: which essay evidence supports each rubric criterion?
            query = rubric_hidden + trait_vec.unsqueeze(1)
            attn_out, _ = self.cross_attention(
                query=query,
                key=essay_rep,
                value=essay_rep,
                key_padding_mask=~essay_mask_rep.bool(),
                need_weights=False,
            )
            # attn_out shape: [B*T, rubric_len, H], so pool using rubric_mask.
            rubric_context = self.pool_tokens(attn_out, rubric_mask, which="context")  # [B*T, H]

        else:
            raise ValueError(f"Unsupported cross_attn_direction: {self.cross_attn_direction}")

        # This is the key design: cross-attention changes the regressor input.
        fused = self.fusion_norm(
            essay_pooled_rep
            + self.rubric_fusion_scale * self.fusion_dropout(rubric_context)
            + self.trait_fusion_scale * trait_vec
        )

        preds = self.shared_regressor(
            fused,
            rubric_hidden=rubric_hidden if self.head_rubric_attention else None,
            rubric_mask=rubric_mask if self.head_rubric_attention else None,
        ).reshape(bsz, num_traits)
        return preds


# ---------------------------------------------------------------------
# Evaluation and prediction output
# ---------------------------------------------------------------------

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
            rubric_input_ids = batch["rubric_input_ids"].to(device)
            rubric_attention_mask = batch["rubric_attention_mask"].to(device)
            labels = batch["labels"].to(device)
            label_mask = batch["label_mask"].to(device)
            idxs = batch["idx"].cpu().numpy()
            token_type_ids = batch["token_type_ids"].to(device) if "token_type_ids" in batch else None

            preds = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                rubric_input_ids=rubric_input_ids,
                rubric_attention_mask=rubric_attention_mask,
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

            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            all_masks.append(label_mask.detach().cpu().numpy())
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


# ---------------------------------------------------------------------
# Optimizer and checkpointing
# ---------------------------------------------------------------------

def build_optimizer(model: RubricFusionTraitWiseAESModel, args):
    if args.freeze_essay_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    if not args.unfreeze_rubric_encoder:
        for p in model.rubric_encoder.parameters():
            p.requires_grad = False

    encoder_params = []
    cross_params = []
    regressor_params = []
    rubric_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(p)
        elif name.startswith("rubric_encoder."):
            rubric_params.append(p)
        elif name.startswith("shared_regressor."):
            regressor_params.append(p)
        elif any(name.startswith(prefix) for prefix in [
            "cross_attention.",
            "fusion_norm.",
            "trait_embeddings.",
            "rubric_projection.",
        ]) or name in {"rubric_fusion_scale", "trait_fusion_scale"}:
            cross_params.append(p)
        else:
            other_params.append(p)

    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": args.lr, "weight_decay": args.weight_decay})
    if cross_params:
        groups.append({"params": cross_params, "lr": args.cross_attn_lr, "weight_decay": args.weight_decay})
    if regressor_params:
        groups.append({"params": regressor_params, "lr": args.regressor_lr, "weight_decay": args.weight_decay})
    if rubric_params:
        groups.append({"params": rubric_params, "lr": args.rubric_lr, "weight_decay": args.weight_decay})
    if other_params:
        groups.append({"params": other_params, "lr": args.cross_attn_lr, "weight_decay": args.weight_decay})

    if not groups:
        raise RuntimeError("No trainable parameters found.")
    return torch.optim.AdamW(groups)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable), "frozen": int(total - trainable)}


def save_checkpoint(
    output_dir: str,
    model: RubricFusionTraitWiseAESModel,
    essay_tokenizer,
    rubric_tokenizer,
    args,
    trait_cols,
    score_ranges,
    prompt_text_map,
    train_metrics,
    dev_metrics,
    param_counts,
):
    ensure_dir(output_dir)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_architecture": "rubric_fusion_traitwise_rubric_attentive_head_base" if args.head_rubric_attention else "rubric_fusion_traitwise_base",
        "model_name": MODEL_NAME,
        "essay_model_name": MODEL_NAME,
        "rubric_model_name": args.rubric_model_name,
        "trait_cols": trait_cols,
        "dropout": args.dropout,
        "max_length": args.max_length,
        "rubric_max_length": args.rubric_max_length,
        "loss_type": args.loss_type,
        "huber_delta": args.huber_delta,
        "heldout_prompt": args.heldout_prompt,
        "cross_attn_heads": args.cross_attn_heads,
        "fusion_init": args.fusion_init,
        "pooling_type": args.pooling_type,
        "cross_attn_direction": args.cross_attn_direction,
        "head_rubric_attention": args.head_rubric_attention,
        "head_attn_heads": args.head_attn_heads if args.head_attn_heads > 0 else args.cross_attn_heads,
        "rubric_text_field": args.rubric_text_field,
        "include_source_excerpt": args.include_source_excerpt,
        "source_excerpt_chars": args.source_excerpt_chars,
        "freeze_rubric_encoder": not args.unfreeze_rubric_encoder,
        "freeze_essay_encoder": args.freeze_essay_encoder,
        "param_counts": param_counts,
    }
    torch.save(ckpt, os.path.join(output_dir, "best_model.pt"))

    # Save both tokenizers/configs. Essay tokenizer at root keeps compatibility with your existing folder pattern.
    essay_tokenizer.save_pretrained(output_dir)
    model.encoder.config.save_pretrained(output_dir)

    rubric_tok_dir = os.path.join(output_dir, "rubric_tokenizer")
    ensure_dir(rubric_tok_dir)
    rubric_tokenizer.save_pretrained(rubric_tok_dir)
    rubric_cfg_dir = os.path.join(output_dir, "rubric_encoder_config")
    ensure_dir(rubric_cfg_dir)
    model.rubric_encoder.config.save_pretrained(rubric_cfg_dir)

    save_json(score_ranges, os.path.join(output_dir, "score_ranges.json"))
    save_json(prompt_text_map, os.path.join(output_dir, "prompt_texts.json"))
    save_json(
        {
            "model_name": MODEL_NAME,
            "model_architecture": "rubric_fusion_traitwise_rubric_attentive_head_base" if args.head_rubric_attention else "rubric_fusion_traitwise_base",
            "heldout_prompt": args.heldout_prompt,
            "trait_cols": trait_cols,
            "prompt_col": args.prompt_col,
            "text_col": args.text_col,
            "id_col": args.id_col,
            "max_length": args.max_length,
            "rubric_max_length": args.rubric_max_length,
            "rubric_model_name": args.rubric_model_name,
            "rubric_text_field": args.rubric_text_field,
            "include_source_excerpt": args.include_source_excerpt,
            "source_excerpt_chars": args.source_excerpt_chars,
            "cross_attn_heads": args.cross_attn_heads,
            "fusion_init": args.fusion_init,
            "pooling_type": args.pooling_type,
            "cross_attn_direction": args.cross_attn_direction,
            "loss_type": args.loss_type,
            "huber_delta": args.huber_delta,
            "round_step": args.round_step,
            "dev_ratio": args.dev_ratio,
            "seed": args.seed,
            "lr": args.lr,
            "cross_attn_lr": args.cross_attn_lr,
            "regressor_lr": args.regressor_lr,
            "rubric_lr": args.rubric_lr,
            "unfreeze_rubric_encoder": args.unfreeze_rubric_encoder,
            "freeze_essay_encoder": args.freeze_essay_encoder,
            "param_counts": param_counts,
        },
        os.path.join(output_dir, "training_config.json"),
    )

    # Keep a copy of the metadata used for this run if available.
    try:
        shutil.copyfile(args.prompt_meta_json, os.path.join(output_dir, "prompt_meta_used.json"))
    except Exception as exc:
        print(f"Warning: could not copy prompt meta JSON: {exc}", flush=True)

    save_json(train_metrics, os.path.join(output_dir, "last_train_metrics.json"))
    save_json(dev_metrics, os.path.join(output_dir, "best_dev_metrics.json"))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(f"Essay/base encoder: {MODEL_NAME}", flush=True)
    print(f"Rubric encoder: {args.rubric_model_name}", flush=True)
    print(f"Pooling type: {args.pooling_type}", flush=True)
    print(f"Cross-attention direction: {args.cross_attn_direction}", flush=True)
    head_attn_heads = args.head_attn_heads if args.head_attn_heads > 0 else args.cross_attn_heads
    print(f"Head rubric attention: {args.head_rubric_attention} (heads={head_attn_heads})", flush=True)

    df = pd.read_csv(args.data_path, sep=args.sep)
    df[args.prompt_col] = df[args.prompt_col].apply(normalize_prompt_id)

    trait_cols = TRAIT_COLUMNS
    print("Trait columns:", trait_cols, flush=True)

    for trait in trait_cols:
        if trait in df.columns:
            df[trait] = pd.to_numeric(df[trait], errors="coerce")

    heldout_prompt = normalize_prompt_id(args.heldout_prompt)
    args.heldout_prompt = heldout_prompt
    all_prompts = sorted(df[args.prompt_col].astype(str).unique().tolist())

    if heldout_prompt not in all_prompts:
        raise ValueError(f"Held-out prompt {heldout_prompt} not found in dataset. Found: {all_prompts}")

    source_df = df[df[args.prompt_col] != heldout_prompt].copy().reset_index(drop=True)
    target_df = df[df[args.prompt_col] == heldout_prompt].copy().reset_index(drop=True)

    train_df, dev_df = split_source_by_prompt(
        df_source=source_df,
        prompt_col=args.prompt_col,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )

    print(f"All prompts      : {all_prompts}", flush=True)
    print(f"Held-out prompt  : {heldout_prompt}", flush=True)
    print(f"Source train size: {len(train_df)}", flush=True)
    print(f"Source dev size  : {len(dev_df)}", flush=True)
    print(f"Held-out size    : {len(target_df)}", flush=True)

    prompt_text_map = build_prompt_text_map()
    prompt_meta = load_prompt_meta(args.prompt_meta_json)
    score_ranges = build_score_ranges_from_hardcoded()
    global_trait_fallback = build_global_trait_fallback(df, trait_cols)

    if args.save_split_files:
        split_dir = os.path.join(args.output_dir, "splits")
        ensure_dir(split_dir)
        train_df.to_csv(os.path.join(split_dir, "source_train.tsv"), sep="\t", index=False)
        dev_df.to_csv(os.path.join(split_dir, "source_dev.tsv"), sep="\t", index=False)
        target_df.to_csv(os.path.join(split_dir, f"heldout_prompt_{heldout_prompt}.tsv"), sep="\t", index=False)

    essay_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    rubric_tokenizer = AutoTokenizer.from_pretrained(args.rubric_model_name, use_fast=True)

    dataset_kwargs = dict(
        essay_tokenizer=essay_tokenizer,
        rubric_tokenizer=rubric_tokenizer,
        trait_cols=trait_cols,
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
    )

    train_dataset = RubricAwareAESDataset(df=train_df, **dataset_kwargs)
    dev_dataset = RubricAwareAESDataset(df=dev_df, **dataset_kwargs)
    target_dataset = RubricAwareAESDataset(df=target_df, **dataset_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = RubricFusionTraitWiseAESModel(
        essay_model_name=MODEL_NAME,
        rubric_model_name=args.rubric_model_name,
        num_traits=len(trait_cols),
        dropout=args.dropout,
        cross_attn_heads=args.cross_attn_heads,
        fusion_init=args.fusion_init,
        freeze_rubric_encoder=not args.unfreeze_rubric_encoder,
        pooling_type=args.pooling_type,
        cross_attn_direction=args.cross_attn_direction,
        head_rubric_attention=args.head_rubric_attention,
        head_attn_heads=head_attn_heads,
    ).to(device)

    optimizer = build_optimizer(model, args)
    param_counts = count_parameters(model)
    print(f"Trainable params: {param_counts['trainable']:,} / {param_counts['total']:,}", flush=True)

    total_update_steps = max(1, math.ceil(len(train_loader) / args.grad_accum_steps) * args.num_epochs)
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    amp_enabled = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_dev_qwk = -1e9
    best_epoch = -1
    early_stop_counter = 0
    last_train_metrics = {}
    history = []

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        if not args.unfreeze_rubric_encoder:
            model.rubric_encoder.eval()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        num_steps = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}", leave=True)

        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            rubric_input_ids = batch["rubric_input_ids"].to(device)
            rubric_attention_mask = batch["rubric_attention_mask"].to(device)
            labels = batch["labels"].to(device)
            label_mask = batch["label_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device) if "token_type_ids" in batch else None

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                preds = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    rubric_input_ids=rubric_input_ids,
                    rubric_attention_mask=rubric_attention_mask,
                    token_type_ids=token_type_ids,
                )
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running_loss += loss.item() * args.grad_accum_steps
            num_steps += 1
            progress.set_postfix(loss=f"{running_loss / max(num_steps, 1):.4f}")

        train_loss = running_loss / max(num_steps, 1)
        last_train_metrics = {"loss": train_loss}

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
            "dev_loss": dev_metrics["loss"],
            "dev_mean_qwk": dev_metrics["mean_qwk"],
            "dev_mean_rmse": dev_metrics["mean_rmse"],
        })
        save_json({"history": history}, os.path.join(args.output_dir, "training_history.json"))

        print()
        print(f"Epoch {epoch} finished", flush=True)
        print(f"Train loss: {train_loss:.6f}", flush=True)
        print(format_metrics_for_print("Source Dev", dev_metrics), flush=True)
        print()

        dev_qwk = dev_metrics["mean_qwk"]
        improved = not math.isnan(dev_qwk) and dev_qwk > best_dev_qwk

        if improved:
            best_dev_qwk = dev_qwk
            best_epoch = epoch
            early_stop_counter = 0
            best_dir = os.path.join(args.output_dir, "best_checkpoint")
            param_counts = count_parameters(model)
            save_checkpoint(
                output_dir=best_dir,
                model=model,
                essay_tokenizer=essay_tokenizer,
                rubric_tokenizer=rubric_tokenizer,
                args=args,
                trait_cols=trait_cols,
                score_ranges=score_ranges,
                prompt_text_map=prompt_text_map,
                train_metrics=last_train_metrics,
                dev_metrics=dev_metrics,
                param_counts=param_counts,
            )
            print(f"Saved new best checkpoint to: {best_dir}", flush=True)
        else:
            early_stop_counter += 1
            print(f"No improvement. Early stop counter: {early_stop_counter}/{args.patience}", flush=True)

        if early_stop_counter >= args.patience:
            print("Early stopping triggered.", flush=True)
            break

    print(f"Best epoch: {best_epoch}", flush=True)
    print(f"Best source-dev mean QWK: {best_dev_qwk:.6f}", flush=True)

    best_ckpt_path = os.path.join(args.output_dir, "best_checkpoint", "best_model.pt")
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(device)

    final_dev_metrics = evaluate_rubric_model(
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
    save_json(final_dev_metrics, os.path.join(args.output_dir, "final_source_dev_metrics.json"))
    print(format_metrics_for_print("Final Source Dev", final_dev_metrics), flush=True)

    if args.run_zero_shot_eval and len(target_dataset) > 0:
        zero_shot_metrics = evaluate_rubric_model(
            model=model,
            dataloader=target_loader,
            dataset=target_dataset,
            trait_cols=trait_cols,
            score_ranges=score_ranges,
            global_trait_fallback=global_trait_fallback,
            device=device,
            round_step=args.round_step,
            loss_type=args.loss_type,
            huber_delta=args.huber_delta,
        )
        save_json(
            zero_shot_metrics,
            os.path.join(args.output_dir, f"zero_shot_heldout_{heldout_prompt}_metrics.json"),
        )
        print()
        print(format_metrics_for_print(f"Zero-shot Held-out Prompt {heldout_prompt}", zero_shot_metrics), flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
