import os
import re
import json
import copy
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import cohen_kappa_score
from transformers import (
    set_seed,
    T5Tokenizer,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model


# =========================================================
# HARD-CODED PROMPT TEXTS
# =========================================================

PROMPT_DESCRIPTIONS = {
    "vuw1": "Should 8-12 year old children use mobile phones?",
    "vuw2": "Should cigarette manufacturers compensate people who develop cancer from cigarette smoking?",
}


# =========================================================
# CONSTANTS
# =========================================================

TARGET_TRAITS = ["ideas", "flow", "coherence", "vocab", "grammar"]

TRAIT_TO_META_NAME = {
    "ideas": "content",
    "flow": "sentence_fluency",
    "coherence": "organization",
    "vocab": "word_choice",
    "grammar": "conventions",
}

TRAIT_RUBRICS = {
    "ideas": "quality of ideas, relevance to the prompt, depth of explanation, and supporting details",
    "flow": "sentence flow, readability, smoothness, and natural phrasing",
    "coherence": "logical sequencing of ideas, paragraph structure, transitions, and clear beginning and ending",
    "vocab": "appropriate vocabulary, precision of word usage, and variety of expressions",
    "grammar": "grammar accuracy, spelling, punctuation, capitalization, and sentence correctness",
}


# =========================================================
# ARGUMENTS
# =========================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for single-fold LoRA adaptation.

    This script assumes the train/dev/test files for one selected fold
    already exist, so it does not create any random split internally.
    """
    parser = argparse.ArgumentParser(
        description="LoRA adaptation for small AES data using one existing fold train/dev/test split."
    )

    parser.add_argument(
        "--folds_dir",
        type=str,
        required=True,
        help="Root directory containing fold subdirectories or fold files.",
    )
    parser.add_argument(
        "--fold_id",
        type=int,
        required=True,
        help="Fold number to run, for example 1 or 2 or 3.",
    )
    parser.add_argument(
        "--train_file_template",
        type=str,
        default="fold_{fold}/train.tsv",
        help="Template relative to folds_dir for the train file.",
    )
    parser.add_argument(
        "--dev_file_template",
        type=str,
        default="fold_{fold}/dev.tsv",
        help="Template relative to folds_dir for the dev file.",
    )
    parser.add_argument(
        "--test_file_template",
        type=str,
        default="fold_{fold}/test.tsv",
        help="Template relative to folds_dir for the test file.",
    )
    parser.add_argument(
        "--sep",
        type=str,
        default="\t",
        help="Column separator used in fold files. Default: tab.",
    )
    parser.add_argument(
        "--no_header",
        action="store_true",
        help="Use when fold files have no header row.",
    )

    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Base/alignment checkpoint path used to rebuild the T5 encoder + regressor model.",
    )
    parser.add_argument(
        "--meta_checkpoint",
        type=str,
        required=True,
        help="Path to meta_model.pt from your FOMAML training.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save outputs for this single fold.",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--max_input_len", type=int, default=768, help="Maximum tokenized input length.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for train/dev/test loaders.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Maximum number of adaptation epochs.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for LoRA adaptation.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for linear scheduler.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience based on dev mean trait QWK.",
    )

    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout.")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,v",
        help="Comma-separated target modules for LoRA, e.g. q,v or q,k,v,o.",
    )
    parser.add_argument(
        "--train_regressor",
        action="store_true",
        help="Also fine-tune the regression head during adaptation.",
    )

    parser.add_argument("--min_score", type=float, default=1.0, help="Minimum score in the target dataset.")
    parser.add_argument("--max_score", type=float, default=6.0, help="Maximum score in the target dataset.")

    return parser.parse_args()


# =========================================================
# GENERAL HELPERS
# =========================================================

def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def setup_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducible runs."""
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_prompt_id(essay_set) -> str:
    """
    Normalize a prompt id into lowercase string form.

    Examples:
    - "VUW1" -> "vuw1"
    - " vuw2 " -> "vuw2"
    """
    return str(essay_set).strip().lower()


def get_prompt_description(essay_set) -> str:
    """
    Return the hardcoded prompt text for a given essay_set.

    This version expects essay sets like:
    - vuw1
    - vuw2

    Add more entries to PROMPT_DESCRIPTIONS when needed.
    """
    normalized = normalize_prompt_id(essay_set)
    if normalized not in PROMPT_DESCRIPTIONS:
        raise KeyError(
            f"Prompt id '{essay_set}' was not found in PROMPT_DESCRIPTIONS. "
            f"Available ids: {sorted(PROMPT_DESCRIPTIONS.keys())}"
        )
    return PROMPT_DESCRIPTIONS[normalized]


def resolve_fold_file(folds_dir: str, template: str, fold_id: int) -> str:
    """
    Resolve a fold-specific file path from a template.

    Example template:
        fold_{fold}/train.tsv
    """
    rel_path = template.format(fold=fold_id)
    return os.path.join(folds_dir, rel_path)


def load_table(path: str, sep: str, no_header: bool) -> pd.DataFrame:
    """
    Read a train/dev/test table and assign expected column names when header is absent.

    Supported no-header formats:
    - 8 columns: essay_id, essay_set, essay, ideas, flow, coherence, vocab, grammar
    - 9 columns: essay_id, essay_set, essay, ideas, flow, coherence, vocab, grammar, overall_score
    """
    if no_header:
        df = pd.read_csv(path, sep=sep, header=None)
        if df.shape[1] == 8:
            df.columns = ["id", "essay_set", "essay", "ideas", "flow", "coherence", "vocab", "grammar"]
        elif df.shape[1] == 9:
            df.columns = ["id", "essay_set", "essay", "ideas", "flow", "coherence", "vocab", "grammar", "overall_score"]
        else:
            raise ValueError(
                f"Unexpected number of columns in {path}. "
                "Expected 8 or 9 columns for --no_header mode."
            )
        return df

    return pd.read_csv(path, sep=sep)


def normalize_score(score: float, min_score: float, max_score: float) -> float:
    """Normalize a raw score into the [0, 1] interval."""
    if max_score == min_score:
        return 0.0
    return (score - min_score) / (max_score - min_score)


def denormalize_score(score_norm: float, min_score: float, max_score: float) -> float:
    """Convert a normalized score back to the original score range."""
    return score_norm * (max_score - min_score) + min_score


def restore_valid_score(score_norm: float, min_score: float, max_score: float) -> int:
    """
    Convert a normalized score into a valid integer rubric score.

    Steps:
    1. Denormalize
    2. Clamp to valid range
    3. Round to nearest integer
    """
    raw = denormalize_score(score_norm, min_score, max_score)
    raw = max(min_score, min(max_score, raw))
    return int(round(raw))


def qwk_safe(y_true: List[int], y_pred: List[int]) -> float:
    """Compute Quadratic Weighted Kappa safely, returning 0.0 if the result is NaN."""
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    if pd.isna(qwk):
        return 0.0
    return float(qwk)


def build_input_text(
    essay_text: str,
    prompt_name: str,
    prompt_text: str,
    trait: str,
    min_score: float,
    max_score: float,
) -> str:
    """
    Build the trait-specific scoring prompt used as model input.

    This keeps the trait-conditioned prompting style from the earlier setup,
    while mapping the target dataset traits to the ASAP-style trait names.
    """
    rubric = TRAIT_RUBRICS[trait]
    return (
        "Score this essay.\n\n"
        f"Prompt name: {prompt_name}\n"
        f"Prompt: {prompt_text}\n"
        f"Trait: {trait}\n"
        f"Trait meaning in source model: {TRAIT_TO_META_NAME[trait]}\n"
        f"Rubric: {rubric}\n"
        f"Score range: {min_score:g} to {max_score:g}\n\n"
        f"Essay:\n{essay_text}"
    )


def prepare_target_dataframe(
    df_wide: pd.DataFrame,
    min_score: float,
    max_score: float,
) -> pd.DataFrame:
    """
    Convert one split from wide essay-level format to long trait-level format.

    Each essay becomes 5 rows, one row for each target trait.
    Prompt text is looked up separately for each row using the hardcoded
    PROMPT_DESCRIPTIONS dictionary and that row's essay_set.
    """
    required = ["essay_id", "essay_set", "essay"] + TARGET_TRAITS
    missing = [c for c in required if c not in df_wide.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_wide = df_wide.copy()

    df_wide["essay_set"] = df_wide["essay_set"].apply(normalize_prompt_id)

    for trait in TARGET_TRAITS:
        df_wide[trait] = df_wide[trait].astype(float)

    if "overall" not in df_wide.columns:
        df_wide["overall"] = df_wide[TARGET_TRAITS].mean(axis=1).round().astype(int)

    records = []
    for _, row in df_wide.iterrows():
        essay_id = row["essay_id"]
        essay_text = str(row["essay"])
        essay_set = str(row["essay_set"])
        prompt_text = get_prompt_description(essay_set)
        overall_true = int(round(float(row["overall"])))

        for trait in TARGET_TRAITS:
            raw_score = float(row[trait])
            score_norm = normalize_score(raw_score, min_score, max_score)

            input_text = build_input_text(
                essay_text=essay_text,
                prompt_name=essay_set,
                prompt_text=prompt_text,
                trait=trait,
                min_score=min_score,
                max_score=max_score,
            )

            records.append(
                {
                    "essay_id": essay_id,
                    "essay_set": essay_set,
                    "essay": essay_text,
                    "trait": trait,
                    "raw_score": raw_score,
                    "score_norm": score_norm,
                    "min_score": min_score,
                    "max_score": max_score,
                    "overall_true": overall_true,
                    "input_text": input_text,
                }
            )

    return pd.DataFrame(records)


# =========================================================
# MODEL
# =========================================================

class T5EncoderRegressor(nn.Module):
    """
    T5 encoder followed by a scalar regression head.

    This matches your earlier setup:
    encoder output -> mean pooling -> regression head -> sigmoid.
    """

    def __init__(self, model_path: str):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_path)
        hidden_size = self.t5.config.d_model
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run the encoder, mean-pool valid token states, and predict one normalized score."""
        enc_out = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = enc_out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        pred = self.regressor(pooled)
        pred = torch.sigmoid(pred)
        return pred


def load_meta_weights(model: T5EncoderRegressor, meta_checkpoint: str, device: str) -> None:
    """Load the meta-trained model weights from meta_model.pt into the adaptation model."""
    ckpt = torch.load(meta_checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"Loaded meta checkpoint: {meta_checkpoint}", flush=True)
    print(f"Missing keys after loading: {len(missing)}", flush=True)
    print(f"Unexpected keys after loading: {len(unexpected)}", flush=True)


def apply_lora(model: T5EncoderRegressor, args: argparse.Namespace) -> None:
    """Attach LoRA modules to the T5 backbone and optionally unfreeze the regression head."""
    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )

    model.t5 = get_peft_model(model.t5, lora_config)

    for p in model.regressor.parameters():
        p.requires_grad = bool(args.train_regressor)

    print("LoRA applied.", flush=True)
    model.t5.print_trainable_parameters()


# =========================================================
# DATASET / COLLATOR
# =========================================================

class AESLongDataset(Dataset):
    """PyTorch dataset over the long-format trait-level dataframe."""

    def __init__(self, df_long: pd.DataFrame):
        self.df = df_long.reset_index(drop=True)

    def __len__(self) -> int:
        """Return the number of trait-level instances."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        """Return one trait-level example with metadata needed for evaluation."""
        row = self.df.iloc[idx]
        return {
            "essay_id": row["essay_id"],
            "essay_set": row["essay_set"],
            "essay": row["essay"],
            "trait": row["trait"],
            "input_text": row["input_text"],
            "raw_score": float(row["raw_score"]),
            "score_norm": float(row["score_norm"]),
            "min_score": float(row["min_score"]),
            "max_score": float(row["max_score"]),
            "overall_true": int(row["overall_true"]),
        }


@dataclass
class Collator:
    """Batch collator that tokenizes input text and moves tensors to the target device."""

    tokenizer: T5Tokenizer
    max_input_len: int
    device: str

    def __call__(self, batch: List[Dict]) -> Dict:
        """Convert a list of examples into one tokenized batch."""
        texts = [x["input_text"] for x in batch]
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_len,
        )
        return {
            "input_ids": enc["input_ids"].to(self.device),
            "attention_mask": enc["attention_mask"].to(self.device),
            "scores": torch.tensor(
                [x["score_norm"] for x in batch],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(-1),
            "meta": batch,
        }


# =========================================================
# TRAINING / EVALUATION
# =========================================================

def compute_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute mean squared error on normalized scores."""
    return F.mse_loss(preds, targets)


def run_epoch(
    model: T5EncoderRegressor,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    max_grad_norm: float,
) -> float:
    """Run one full training epoch and return the mean training loss."""
    model.train()
    total_loss = 0.0

    for batch in data_loader:
        optimizer.zero_grad()
        preds = model(batch["input_ids"], batch["attention_mask"])
        loss = compute_loss(preds, batch["scores"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / max(len(data_loader), 1)


def build_essay_level_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse trait-level predictions into one row per essay.

    The overall score is computed as the rounded mean of the 5 trait scores.
    """
    rows = []
    for essay_id, group in pred_df.groupby("essay_id"):
        row = {
            "essay_id": essay_id,
            "essay_set": group["essay_set"].iloc[0],
            "essay": group["essay"].iloc[0],
        }

        true_vals = []
        pred_vals = []

        for trait in TARGET_TRAITS:
            g = group[group["trait"] == trait]
            if len(g) == 0:
                row[f"true_{trait}"] = np.nan
                row[f"pred_{trait}"] = np.nan
            else:
                true_score = int(g["true_score"].iloc[0])
                pred_score = int(g["pred_score"].iloc[0])
                row[f"true_{trait}"] = true_score
                row[f"pred_{trait}"] = pred_score
                true_vals.append(true_score)
                pred_vals.append(pred_score)

        row["true_overall"] = int(round(np.mean(true_vals))) if true_vals else np.nan
        row["pred_overall"] = int(round(np.mean(pred_vals))) if pred_vals else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate(model: T5EncoderRegressor, data_loader: DataLoader) -> Tuple[Dict, pd.DataFrame]:
    """
    Evaluate the model and return metrics plus essay-level predictions.

    Returned metrics include:
    - loss
    - MAE / RMSE
    - trait-wise QWK
    - mean trait QWK
    - overall QWK
    """
    model.eval()
    losses = []
    records = []

    with torch.no_grad():
        for batch in data_loader:
            preds = model(batch["input_ids"], batch["attention_mask"])
            loss = compute_loss(preds, batch["scores"])
            losses.append(loss.item())

            pred_norm = preds.squeeze(-1).detach().cpu().numpy().tolist()
            true_norm = batch["scores"].squeeze(-1).detach().cpu().numpy().tolist()

            for meta, p_norm, t_norm in zip(batch["meta"], pred_norm, true_norm):
                pred_score = restore_valid_score(p_norm, meta["min_score"], meta["max_score"])
                true_score = restore_valid_score(t_norm, meta["min_score"], meta["max_score"])

                records.append(
                    {
                        "essay_id": meta["essay_id"],
                        "essay_set": meta["essay_set"],
                        "trait": meta["trait"],
                        "essay": meta["essay"],
                        "true_score": true_score,
                        "pred_score": pred_score,
                    }
                )

    pred_df = pd.DataFrame(records)

    metrics = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "trait_qwk": {trait: 0.0 for trait in TARGET_TRAITS},
        "mean_trait_qwk": 0.0,
        "overall_qwk": 0.0,
        "mae": 0.0,
        "rmse": 0.0,
    }

    if pred_df.empty:
        return metrics, pd.DataFrame()

    trait_qwk = {}
    for trait in TARGET_TRAITS:
        part = pred_df[pred_df["trait"] == trait]
        trait_qwk[trait] = qwk_safe(
            part["true_score"].astype(int).tolist(),
            part["pred_score"].astype(int).tolist(),
        ) if len(part) > 0 else 0.0

    metrics["trait_qwk"] = trait_qwk
    metrics["mean_trait_qwk"] = float(np.mean(list(trait_qwk.values())))
    metrics["mae"] = float(np.mean(np.abs(pred_df["true_score"] - pred_df["pred_score"])))
    metrics["rmse"] = float(np.sqrt(np.mean((pred_df["true_score"] - pred_df["pred_score"]) ** 2)))

    essay_df = build_essay_level_predictions(pred_df)
    if not essay_df.empty:
        metrics["overall_qwk"] = qwk_safe(
            essay_df["true_overall"].astype(int).tolist(),
            essay_df["pred_overall"].astype(int).tolist(),
        )

    return metrics, essay_df


def build_dataloader(
    df_long: pd.DataFrame,
    tokenizer: T5Tokenizer,
    batch_size: int,
    max_input_len: int,
    device: str,
    shuffle: bool,
) -> DataLoader:
    """Create a DataLoader for one split."""
    return DataLoader(
        AESLongDataset(df_long),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=Collator(tokenizer, max_input_len, device),
    )


def summarize_result(test_metrics: Dict, best_epoch: int, best_dev_qwk: float) -> Dict:
    """Build a compact summary dictionary for the selected fold."""
    return {
        "best_epoch": best_epoch,
        "best_dev_mean_trait_qwk": float(best_dev_qwk),
        "test_loss": float(test_metrics["loss"]),
        "test_trait_qwk": {k: float(v) for k, v in test_metrics["trait_qwk"].items()},
        "test_mean_trait_qwk": float(test_metrics["mean_trait_qwk"]),
        "test_overall_qwk": float(test_metrics["overall_qwk"]),
        "test_mae": float(test_metrics["mae"]),
        "test_rmse": float(test_metrics["rmse"]),
    }


# =========================================================
# SINGLE FOLD RUNNER
# =========================================================

def run_single_fold(
    args: argparse.Namespace,
    tokenizer: T5Tokenizer,
    device: str,
) -> Dict:
    """
    Train and evaluate LoRA adaptation on one selected fold.

    This function:
    1. Loads train/dev/test files for the selected fold
    2. Converts them to long trait-wise format
    3. Loads the meta-trained checkpoint
    4. Applies LoRA
    5. Trains using the train split only
    6. Selects the best model using the dev split
    7. Evaluates on the test split
    """
    train_path = resolve_fold_file(args.folds_dir, args.train_file_template, args.fold_id)
    dev_path = resolve_fold_file(args.folds_dir, args.dev_file_template, args.fold_id)
    test_path = resolve_fold_file(args.folds_dir, args.test_file_template, args.fold_id)

    for path in [train_path, dev_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fold {args.fold_id} file not found: {path}")

    print("\n" + "=" * 80, flush=True)
    print(f"Running fold {args.fold_id}", flush=True)
    print(f"Train file: {train_path}", flush=True)
    print(f"Dev file:   {dev_path}", flush=True)
    print(f"Test file:  {test_path}", flush=True)
    print("=" * 80, flush=True)

    train_wide = load_table(train_path, args.sep, args.no_header)
    dev_wide = load_table(dev_path, args.sep, args.no_header)
    test_wide = load_table(test_path, args.sep, args.no_header)

    split_info = {
        "fold": args.fold_id,
        "n_train_essays": int(len(train_wide)),
        "n_dev_essays": int(len(dev_wide)),
        "n_test_essays": int(len(test_wide)),
        "train_essay_sets": sorted(train_wide["essay_set"].astype(str).str.strip().str.lower().unique().tolist()),
        "dev_essay_sets": sorted(dev_wide["essay_set"].astype(str).str.strip().str.lower().unique().tolist()),
        "test_essay_sets": sorted(test_wide["essay_set"].astype(str).str.strip().str.lower().unique().tolist()),
        "available_prompt_descriptions": sorted(PROMPT_DESCRIPTIONS.keys()),
        "traits": TARGET_TRAITS,
    }
    with open(os.path.join(args.output_dir, "split_info.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    train_long = prepare_target_dataframe(train_wide, args.min_score, args.max_score)
    dev_long = prepare_target_dataframe(dev_wide, args.min_score, args.max_score)
    test_long = prepare_target_dataframe(test_wide, args.min_score, args.max_score)

    train_loader = build_dataloader(train_long, tokenizer, args.batch_size, args.max_input_len, device, shuffle=True)
    dev_loader = build_dataloader(dev_long, tokenizer, args.batch_size, args.max_input_len, device, shuffle=False)
    test_loader = build_dataloader(test_long, tokenizer, args.batch_size, args.max_input_len, device, shuffle=False)

    model = T5EncoderRegressor(args.base_model_path)
    load_meta_weights(model, args.meta_checkpoint, device)
    apply_lora(model, args)
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = max(len(train_loader) * args.num_epochs, 1)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = []
    best_state = None
    best_epoch = -1
    best_dev_qwk = -1e9
    patience_counter = 0

    for epoch in range(1, args.num_epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scheduler, args.max_grad_norm)
        dev_metrics, _ = evaluate(model, dev_loader)

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "dev_loss": float(dev_metrics["loss"]),
            "dev_trait_qwk": {k: float(v) for k, v in dev_metrics["trait_qwk"].items()},
            "dev_mean_trait_qwk": float(dev_metrics["mean_trait_qwk"]),
            "dev_overall_qwk": float(dev_metrics["overall_qwk"]),
        }
        history.append(epoch_record)

        print(f"\nFold {args.fold_id} | Epoch {epoch}", flush=True)
        print(json.dumps(epoch_record, indent=2), flush=True)

        if dev_metrics["mean_trait_qwk"] > best_dev_qwk:
            best_dev_qwk = dev_metrics["mean_trait_qwk"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0

            torch.save(best_state, os.path.join(args.output_dir, "best_full_model_state.pt"))

            adapter_dir = os.path.join(args.output_dir, "best_lora_adapter")
            ensure_dir(adapter_dir)
            model.t5.save_pretrained(adapter_dir)
            tokenizer.save_pretrained(adapter_dir)
            torch.save(model.regressor.state_dict(), os.path.join(adapter_dir, "regressor.pt"))
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping on fold {args.fold_id} at epoch {epoch}", flush=True)
            break

    with open(os.path.join(args.output_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    if best_state is None:
        raise RuntimeError(f"No best model state saved for fold {args.fold_id}.")

    model.load_state_dict(best_state)
    test_metrics, test_pred_df = evaluate(model, test_loader)
    result = summarize_result(test_metrics, best_epoch, best_dev_qwk)
    result["fold"] = args.fold_id
    result["prompt_descriptions"] = PROMPT_DESCRIPTIONS

    with open(os.path.join(args.output_dir, "final_test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    test_pred_df.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)

    print(f"\nFold {args.fold_id} final test results", flush=True)
    for trait in TARGET_TRAITS:
        print(f"QWK [{trait}]: {result['test_trait_qwk'][trait]:.4f}", flush=True)
    print(f"Mean trait QWK: {result['test_mean_trait_qwk']:.4f}", flush=True)
    print(f"Overall QWK: {result['test_overall_qwk']:.4f}", flush=True)
    print(f"MAE: {result['test_mae']:.4f}", flush=True)
    print(f"RMSE: {result['test_rmse']:.4f}", flush=True)

    return result


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    """Run LoRA adaptation for one selected fold only and save the outputs."""
    args = parse_args()
    ensure_dir(args.output_dir)

    fold_seed = args.seed + args.fold_id
    setup_seed(fold_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    print(f"Running fold: {args.fold_id}", flush=True)
    print(f"Available prompt ids in code: {sorted(PROMPT_DESCRIPTIONS.keys())}", flush=True)

    tokenizer = T5Tokenizer.from_pretrained(args.base_model_path)

    result = run_single_fold(
        args=args,
        tokenizer=tokenizer,
        device=device,
    )

    summary = {
        "fold_result": result,
        "prompt_descriptions": PROMPT_DESCRIPTIONS,
    }
    with open(os.path.join(args.output_dir, "single_fold_result.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "#" * 80, flush=True)
    print("FINAL RESULT FOR SELECTED FOLD", flush=True)
    print(f"Fold: {args.fold_id}", flush=True)
    for trait in TARGET_TRAITS:
        print(f"QWK [{trait}]: {result['test_trait_qwk'][trait]:.4f}", flush=True)
    print(f"Mean trait QWK: {result['test_mean_trait_qwk']:.4f}", flush=True)
    print(f"Overall QWK: {result['test_overall_qwk']:.4f}", flush=True)
    print(f"MAE: {result['test_mae']:.4f}", flush=True)
    print(f"RMSE: {result['test_rmse']:.4f}", flush=True)
    print("#" * 80, flush=True)


if __name__ == "__main__":
    main()