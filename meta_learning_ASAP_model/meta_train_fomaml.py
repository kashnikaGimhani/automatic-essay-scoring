import os
import json
import random
import argparse
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    set_seed,
    T5Tokenizer,
    T5ForConditionalGeneration,
)

try:
    from torch.func import functional_call
except ImportError:
    raise ImportError(
        "This script requires torch.nn.utils.stateless.functional_call. "
        "Please use a compatible PyTorch version."
    )

from sklearn.metrics import cohen_kappa_score

print("Imports completed", flush=True)


# =========================================================
# CONSTANTS
# Reused from your alignment script so prompt descriptions,
# rubric text, and score ranges remain consistent.
# =========================================================

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

TRAIT_RUBRICS = {
    "content": "quality of ideas, relevance to the prompt, depth of explanation, and supporting details",
    "organization": "logical sequencing of ideas, paragraph structure, transitions, and clear beginning and ending",
    "word_choice": "appropriate vocabulary, precision of word usage, and variety of expressions",
    "sentence_fluency": "sentence flow, readability, smoothness, and natural phrasing",
    "conventions": "grammar accuracy, spelling, punctuation, capitalization, and sentence correctness",
    "prompt_adherence": "how well the essay addresses the prompt and stays on topic",
    "language": "overall language control, correctness, and clarity of expression",
    "narrativity": "storytelling quality, narrative development, sequencing of events, and engagement",
    "style": "effectiveness of expression, tone, and rhetorical quality",
    "voice": "writer presence, individuality, and distinctiveness of expression",
    "overall": "overall quality of the essay considering content, structure, language, and effectiveness",
}

SCORE_RANGES: Dict[Tuple[int, str], Tuple[float, float]] = {
    (1, "content"): (1, 6),
    (1, "organization"): (1, 6),
    (1, "word_choice"): (1, 6),
    (1, "sentence_fluency"): (1, 6),
    (1, "conventions"): (1, 6),

    (2, "content"): (1, 6),
    (2, "organization"): (1, 6),
    (2, "word_choice"): (1, 6),
    (2, "sentence_fluency"): (1, 6),
    (2, "conventions"): (1, 6),

    (3, "content"): (0, 3),
    (3, "prompt_adherence"): (0, 3),
    (3, "language"): (0, 3),
    (3, "narrativity"): (0, 3),

    (4, "content"): (0, 3),
    (4, "prompt_adherence"): (0, 3),
    (4, "language"): (0, 3),
    (4, "narrativity"): (0, 3),

    (5, "content"): (0, 4),
    (5, "prompt_adherence"): (0, 8),
    (5, "language"): (0, 8),
    (5, "narrativity"): (0, 8),

    (6, "content"): (0, 8),
    (6, "prompt_adherence"): (0, 4),
    (6, "language"): (0, 4),
    (6, "narrativity"): (0, 4),

    (7, "content"): (0, 6),
    (7, "organization"): (0, 6),
    (7, "style"): (0, 6),
    (7, "conventions"): (0, 6),

    (8, "content"): (2, 12),
    (8, "organization"): (2, 12),
    (8, "voice"): (2, 12),
    (8, "word_choice"): (2, 12),
    (8, "sentence_fluency"): (2, 12),
    (8, "conventions"): (2, 12),
}

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


# =========================================================
# ARGUMENTS
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="True first-order MAML style meta-training for AES."
    )

    parser.add_argument("--train_file", type=str, required=True, help="Path to ASAP TSV/CSV file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to alignment finetuned checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_input_len", type=int, default=768)

    parser.add_argument("--train_prompts", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--val_prompts", type=int, nargs="+", default=[8])
    parser.add_argument("--test_prompts", type=int, nargs="+", default=[7])

    parser.add_argument("--max_samples_per_group_train", type=int, default=300)
    parser.add_argument("--max_samples_per_group_val", type=int, default=100)
    parser.add_argument("--max_samples_per_group_test",type=int,default=None,help="Limit number of samples per (prompt, trait) group for test set")

    parser.add_argument("--support_size", type=int, default=8)
    parser.add_argument("--query_size", type=int, default=16)
    parser.add_argument("--tasks_per_meta_batch", type=int, default=4)
    parser.add_argument("--meta_steps", type=int, default=1000)

    parser.add_argument("--inner_steps", type=int, default=1)
    parser.add_argument("--inner_lr", type=float, default=1e-3)
    parser.add_argument("--meta_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--trainable_mode", type=str, default="head", choices=["head", "last_k", "all"])
    parser.add_argument("--unfreeze_last_k", type=int, default=2)

    parser.add_argument("--val_every", type=int, default=100)
    parser.add_argument("--val_episodes", type=int, default=100)
    parser.add_argument("--test_episodes", type=int, default=200)
    parser.add_argument("--print_every", type=int, default=20)

    return parser.parse_args()


# =========================================================
# HELPERS
# =========================================================

def setup_seed(seed: int):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_table(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep="\t")


def normalize_score(score: float, min_score: float, max_score: float) -> float:
    if max_score == min_score:
        return 0.0
    return (score - min_score) / (max_score - min_score)


def infer_score_ranges(df: pd.DataFrame, trait_columns: List[str]) -> Dict[Tuple[int, str], Tuple[float, float]]: #need to use the predefined min max score ranges
    score_ranges = {}
    for essay_set in sorted(df["essay_set"].dropna().unique()):
        df_set = df[df["essay_set"] == essay_set]
        for trait in trait_columns:
            if trait in df_set.columns:
                values = df_set[trait].dropna().astype(float)
                if len(values) > 0:
                    score_ranges[(int(essay_set), trait)] = (float(values.min()), float(values.max()))
    return score_ranges


def get_score_range(
    essay_set: int,
    trait: str,
    inferred_ranges: Dict[Tuple[int, str], Tuple[float, float]],
) -> Tuple[float, float]:
    if (essay_set, trait) in SCORE_RANGES:
        return SCORE_RANGES[(essay_set, trait)]
    if (essay_set, trait) in inferred_ranges:
        return inferred_ranges[(essay_set, trait)]
    raise ValueError(f"No score range found for essay_set={essay_set}, trait={trait}")


def build_input_text(
    essay_text: str,
    essay_set: int,
    trait: str,
    min_score: float,
    max_score: float,
) -> str:
    prompt_desc = PROMPT_DESCRIPTIONS.get(essay_set, f"Essay set {essay_set}")
    rubric = TRAIT_RUBRICS.get(trait, f"Score the essay for {trait}.")

    return (
        "Score this essay.\n\n"
        f"Prompt: {prompt_desc}\n"
        f"Trait: {trait}\n"
        f"Rubric: {rubric}\n"
        f"Score range: {min_score:g} to {max_score:g}\n\n"
        f"Essay:\n{essay_text}"
    )


def wide_to_long_with_norm(
    df: pd.DataFrame,
    trait_columns: List[str],
    inferred_ranges: Dict[Tuple[int, str], Tuple[float, float]],
) -> pd.DataFrame:
    records = []

    for _, row in df.iterrows():
        essay_id = row["essay_id"]
        essay_set = int(row["essay_set"])
        essay_text = str(row["essay"])

        for trait in trait_columns:
            if trait in df.columns and pd.notna(row[trait]):
                raw_score = float(row[trait])
                min_score, max_score = get_score_range(essay_set, trait, inferred_ranges)
                score_norm = normalize_score(raw_score, min_score, max_score)

                input_text = build_input_text(
                    essay_text=essay_text,
                    essay_set=essay_set,
                    trait=trait,
                    min_score=min_score,
                    max_score=max_score,
                )

                records.append(
                    {
                        "essay_id": essay_id,
                        "essay_set": essay_set,
                        "essay_text": essay_text,
                        "trait": trait,
                        "raw_score": raw_score,
                        "min_score": min_score,
                        "max_score": max_score,
                        "score_norm": score_norm,
                        "input_text": input_text,
                    }
                )

    return pd.DataFrame(records)


def balanced_subset(df_long: pd.DataFrame, max_per_group: int, seed: int) -> pd.DataFrame:
    parts = []
    grouped = df_long.groupby(["essay_set", "trait"])

    for _, group_df in grouped:
        if len(group_df) > max_per_group:
            group_df = group_df.sample(max_per_group, random_state=seed)
        parts.append(group_df)

    if len(parts) == 0:
        raise ValueError("No data available after grouping.")

    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def sample_without_replacement(indices: List[int], n: int) -> List[int]:
    if len(indices) < n:
        raise ValueError(f"Not enough samples. Need {n}, found {len(indices)}")
    return random.sample(indices, n)

def denormalize_score(score_norm: float, min_score: float, max_score: float) -> float:
    """
    Convert normalized score back to original score range.
    """
    return score_norm * (max_score - min_score) + min_score


def restore_valid_score(score_norm: float, min_score: float, max_score: float) -> int:
    """
    Convert normalized score to a valid integer score:
    1. denormalize
    2. clamp to valid range
    3. round to nearest integer
    """
    raw_score = denormalize_score(score_norm, min_score, max_score)
    raw_score = max(min_score, min(max_score, raw_score))
    return int(round(raw_score))


def compute_qwk_from_records(records: List[Dict]) -> Dict[str, float]:
    """
    Compute QWK separately for each (essay_set, trait) group,
    then return mean QWK across groups.
    """
    if len(records) == 0:
        return {
            "mean_qwk": 0.0,
            "group_qwks": {},
        }

    df_tmp = pd.DataFrame(records)

    group_qwks = {}
    qwk_values = []

    for (essay_set, trait), group_df in df_tmp.groupby(["essay_set", "trait"]):
        min_score = float(group_df["min_score"].iloc[0])
        max_score = float(group_df["max_score"].iloc[0])

        y_true = [
            restore_valid_score(v, min_score, max_score)
            for v in group_df["true_norm"].astype(float).tolist()
        ]
        y_pred = [
            restore_valid_score(v, min_score, max_score)
            for v in group_df["pred_norm"].astype(float).tolist()
        ]

        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        if pd.isna(qwk):
            qwk = 0.0

        key = f"prompt_{essay_set}_{trait}"
        group_qwks[key] = float(qwk)
        qwk_values.append(float(qwk))

    mean_qwk = float(np.mean(qwk_values)) if len(qwk_values) > 0 else 0.0

    return {
        "mean_qwk": mean_qwk,
        "group_qwks": group_qwks,
    }


# =========================================================
# MODEL
# T5 encoder + regression head
# This stays the same as your current meta script.
# =========================================================

class T5EncoderRegressor(nn.Module):
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

    def _forward_impl(self, input_ids, attention_mask):
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

    def forward(self, input_ids, attention_mask):
        return self._forward_impl(input_ids, attention_mask)


def set_trainable_params(model: T5EncoderRegressor, mode: str, unfreeze_last_k: int):
    for p in model.parameters():
        p.requires_grad = False

    if mode == "head":
        for p in model.regressor.parameters():
            p.requires_grad = True

    elif mode == "last_k":
        for p in model.regressor.parameters():
            p.requires_grad = True

        blocks = model.t5.encoder.block
        k = min(unfreeze_last_k, len(blocks))
        for block in blocks[-k:]:
            for p in block.parameters():
                p.requires_grad = True

        for p in model.t5.encoder.final_layer_norm.parameters():
            p.requires_grad = True

    elif mode == "all":
        for p in model.t5.encoder.parameters():
            p.requires_grad = True
        for p in model.regressor.parameters():
            p.requires_grad = True

    else:
        raise ValueError(f"Unknown mode: {mode}")


# =========================================================
# EPISODIC DATASET
# Groups long-format rows into tasks:
#   task = (essay_set, trait)
# =========================================================

class EpisodicAESDataset:
    def __init__(
        self,
        df_long: pd.DataFrame,
        tokenizer: T5Tokenizer,
        max_input_len: int,
        support_size: int,
        query_size: int,
        device: str,
    ):
        self.df = df_long.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.support_size = support_size
        self.query_size = query_size
        self.device = device

        self.task_to_indices: Dict[Tuple[int, str], List[int]] = {}

        for idx, row in self.df.iterrows():
            task = (int(row["essay_set"]), str(row["trait"]))
            self.task_to_indices.setdefault(task, []).append(idx)

        self.tasks = [
            task for task, idxs in self.task_to_indices.items()
            if len(idxs) >= (self.support_size + self.query_size)
        ]

        if len(self.tasks) == 0:
            raise ValueError(
                "No tasks have enough examples for support + query split. "
                "Reduce support_size/query_size or inspect your data."
            )

    def sample_task_batch(self, num_tasks: int) -> List[Tuple[int, str]]:
        if len(self.tasks) >= num_tasks:
            return random.sample(self.tasks, num_tasks)
        return random.choices(self.tasks, k=num_tasks)

    def sample_episode(self, task: Tuple[int, str]) -> Dict[str, Dict[str, torch.Tensor]]:
        indices = self.task_to_indices[task]
        chosen = sample_without_replacement(indices, self.support_size + self.query_size)

        support_indices = chosen[:self.support_size]
        query_indices = chosen[self.support_size:]

        support_df = self.df.iloc[support_indices]
        query_df = self.df.iloc[query_indices]

        return {
            "support": self._encode_batch(support_df),
            "query": self._encode_batch(query_df),
        }

    def _encode_batch(self, batch_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        texts = batch_df["input_text"].tolist()
        scores = batch_df["score_norm"].astype(float).tolist()

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
            "scores": torch.tensor(scores, dtype=torch.float32, device=self.device).unsqueeze(-1),

            # metadata for QWK
            "essay_set": batch_df["essay_set"].astype(int).tolist(),
            "trait": batch_df["trait"].astype(str).tolist(),
            "min_score": batch_df["min_score"].astype(float).tolist(),
            "max_score": batch_df["max_score"].astype(float).tolist(),
        }


# =========================================================
# FUNCTIONAL FOMAML HELPERS
# This is the key difference from your approximate version.
# We do NOT deepcopy the model.
# Instead we:
# 1. get trainable parameter dict
# 2. compute support gradients
# 3. create fast weights
# 4. evaluate query loss using fast weights
# =========================================================

def get_trainable_parameter_dict(model: nn.Module) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict(
        (name, param) for name, param in model.named_parameters() if param.requires_grad
    )


def get_buffer_dict(model: nn.Module) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict(model.named_buffers())


def build_full_parameter_dict(
    model: nn.Module,
    fast_weights: "OrderedDict[str, torch.Tensor]",
) -> "OrderedDict[str, torch.Tensor]":
    """
    Build a full parameter dictionary for functional_call:
    - fast weights for trainable params
    - original params for frozen params
    """
    full_params = OrderedDict()
    for name, param in model.named_parameters():
        if name in fast_weights:
            full_params[name] = fast_weights[name]
        else:
            full_params[name] = param
    return full_params


def functional_forward(model, batch, fast_weights=None):
    if fast_weights is None:
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

    full_params = build_full_parameter_dict(model, fast_weights)

    return functional_call(
        model,
        full_params,
        args=(batch["input_ids"], batch["attention_mask"]),
    )


def mse_loss_from_batch(
    model: T5EncoderRegressor,
    batch: Dict[str, torch.Tensor],
    fast_weights: "OrderedDict[str, torch.Tensor]" = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    preds = functional_forward(model, batch, fast_weights=fast_weights)
    loss = F.mse_loss(preds, batch["scores"])
    return loss, preds


def inner_update_fomaml(
    model: T5EncoderRegressor,
    support_batch: Dict[str, torch.Tensor],
    inner_lr: float,
    inner_steps: int,
) -> "OrderedDict[str, torch.Tensor]":
    """
    True first-order style inner-loop update.

    Start from current trainable parameters:
        theta

    Then for each inner step:
        grads = grad(L_support(theta))
        theta' = theta - alpha * grads

    In FOMAML, we use create_graph=False.
    """
    fast_weights = get_trainable_parameter_dict(model)

    for _ in range(inner_steps):
        support_loss, _ = mse_loss_from_batch(
            model=model,
            batch=support_batch,
            fast_weights=fast_weights,
        )

        grads = torch.autograd.grad(
            support_loss,
            list(fast_weights.values()),
            create_graph=False,   # first-order MAML
            retain_graph=False,
            allow_unused=False,
        )

        fast_weights = OrderedDict(
            (name, param - inner_lr * grad)
            for (name, param), grad in zip(fast_weights.items(), grads)
        )

    return fast_weights


# =========================================================
# EPISODIC EVALUATION
# Adapt with fast weights on support, evaluate on query.
# =========================================================

@torch.no_grad()
def simple_metrics_from_lists(preds: List[float], targets: List[float]) -> Dict[str, float]:
    preds = np.array(preds, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))

    return {
        "mse_norm": mse,
        "mae_norm": mae,
    }


def evaluate_episodic(
    model: T5EncoderRegressor,
    episodic_data: EpisodicAESDataset,
    num_episodes: int,
    inner_lr: float,
    inner_steps: int,
) -> Dict[str, float]:
    model.eval()

    losses = []
    all_preds = []
    all_targets = []
    qwk_records = []

    for _ in range(num_episodes):
        task = random.choice(episodic_data.tasks)
        episode = episodic_data.sample_episode(task)

        support_batch = episode["support"]
        query_batch = episode["query"]

        with torch.enable_grad():
            fast_weights = inner_update_fomaml(
                model=model,
                support_batch=support_batch,
                inner_lr=inner_lr,
                inner_steps=inner_steps,
            )

        with torch.no_grad():
            query_loss, query_preds = mse_loss_from_batch(
                model=model,
                batch=query_batch,
                fast_weights=fast_weights,
            )

        losses.append(query_loss.item())

        pred_list = query_preds.squeeze(-1).cpu().numpy().tolist()
        true_list = query_batch["scores"].squeeze(-1).cpu().numpy().tolist()

        all_preds.extend(pred_list)
        all_targets.extend(true_list)

        # collect metadata row-by-row for QWK
        for i in range(len(pred_list)):
            qwk_records.append(
                {
                    "essay_set": int(query_batch["essay_set"][i]),
                    "trait": str(query_batch["trait"][i]),
                    "min_score": float(query_batch["min_score"][i]),
                    "max_score": float(query_batch["max_score"][i]),
                    "true_norm": float(true_list[i]),
                    "pred_norm": float(pred_list[i]),
                }
            )

    all_preds = np.array(all_preds, dtype=np.float32)
    all_targets = np.array(all_targets, dtype=np.float32)

    mse = float(np.mean((all_preds - all_targets) ** 2))
    mae = float(np.mean(np.abs(all_preds - all_targets)))

    qwk_results = compute_qwk_from_records(qwk_records)

    return {
        "loss": float(np.mean(losses)),
        "mse_norm": mse,
        "mae_norm": mae,
        "mean_qwk": qwk_results["mean_qwk"],
    }


# =========================================================
# CHECKPOINT SAVING
# =========================================================

def save_checkpoint(model, tokenizer, save_dir, extra_info=None):
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, "meta_model.pt"))
    tokenizer.save_pretrained(save_dir)

    if extra_info is not None:
        with open(os.path.join(save_dir, "meta_info.json"), "w") as f:
            json.dump(extra_info, f, indent=2)


# =========================================================
# MAIN
# =========================================================

def main():
    print("Entering true FOMAML meta-training script...", flush=True)

    args = parse_args()
    setup_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Device: {device}", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------
    df = load_table(args.train_file)
    print(f"Loaded raw data with {len(df)} rows.", flush=True)

    required_cols = {"essay_id", "essay_set", "essay"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    available_traits = [t for t in TRAIT_COLUMNS if t in df.columns]
    print("Available traits:", available_traits, flush=True)

    inferred_score_ranges = infer_score_ranges(df, available_traits)

    # -----------------------------------------------------
    # Prompt split
    # -----------------------------------------------------
    df_train_pool = df[df["essay_set"].isin(args.train_prompts)].copy()
    df_val_pool = df[df["essay_set"].isin(args.val_prompts)].copy()
    df_test_pool = df[df["essay_set"].isin(args.test_prompts)].copy()

    print("\nPrompt-based split sizes:", flush=True)
    print("Train pool essays:", len(df_train_pool), flush=True)
    print("Val pool essays:", len(df_val_pool), flush=True)
    print("Test pool essays:", len(df_test_pool), flush=True)

    # -----------------------------------------------------
    # Wide -> long
    # -----------------------------------------------------
    df_train_long = wide_to_long_with_norm(df_train_pool, available_traits, inferred_score_ranges)
    df_val_long = wide_to_long_with_norm(df_val_pool, available_traits, inferred_score_ranges)
    df_test_long = wide_to_long_with_norm(df_test_pool, available_traits, inferred_score_ranges)

    print("\nTrait-wise sizes before balancing:", flush=True)
    print("Train long:", len(df_train_long), flush=True)
    print("Val long:", len(df_val_long), flush=True)
    print("Test long:", len(df_test_long), flush=True)

    # -----------------------------------------------------
    # Balance groups
    # -----------------------------------------------------
    df_train_meta = balanced_subset(
        df_train_long,
        max_per_group=args.max_samples_per_group_train,
        seed=args.seed,
    )
    df_val_meta = balanced_subset(
        df_val_long,
        max_per_group=args.max_samples_per_group_val,
        seed=args.seed,
    )
    if args.max_samples_per_group_test is not None:
        df_test_meta = balanced_subset(
            df_test_long,
            max_per_group=args.max_samples_per_group_test,
            seed=args.seed,
        )
    else:
        df_test_meta = df_test_long.reset_index(drop=True)

    print("\nTrait-wise sizes after balancing:", flush=True)
    print("Train meta:", len(df_train_meta), flush=True)
    print("Val meta:", len(df_val_meta), flush=True)
    print("Test meta:", len(df_test_meta), flush=True)

    # -----------------------------------------------------
    # Load tokenizer and model
    # -----------------------------------------------------
    print("Loading tokenizer...", flush=True)
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)

    print("Loading model...", flush=True)
    model = T5EncoderRegressor(args.model_path)
    set_trainable_params(model, args.trainable_mode, args.unfreeze_last_k)
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\nTrainable mode: {args.trainable_mode}", flush=True)
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}", flush=True)
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    meta_optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.meta_lr,
        weight_decay=args.weight_decay,
    )

    # -----------------------------------------------------
    # Episodic datasets
    # -----------------------------------------------------
    train_episodic = EpisodicAESDataset(
        df_long=df_train_meta,
        tokenizer=tokenizer,
        max_input_len=args.max_input_len,
        support_size=args.support_size,
        query_size=args.query_size,
        device=device,
    )
    val_episodic = EpisodicAESDataset(
        df_long=df_val_meta,
        tokenizer=tokenizer,
        max_input_len=args.max_input_len,
        support_size=args.support_size,
        query_size=args.query_size,
        device=device,
    )
    test_episodic = EpisodicAESDataset(
        df_long=df_test_meta,
        tokenizer=tokenizer,
        max_input_len=args.max_input_len,
        support_size=args.support_size,
        query_size=args.query_size,
        device=device,
    )

    print(f"\nTrain tasks: {len(train_episodic.tasks)}", flush=True)
    print(f"Val tasks: {len(val_episodic.tasks)}", flush=True)
    print(f"Test tasks: {len(test_episodic.tasks)}", flush=True)

    # -----------------------------------------------------
    # Meta-training loop
    # -----------------------------------------------------
    best_val_mse = float("inf")
    best_step = -1

    print("\nStarting true FOMAML meta-training...\n", flush=True)

    for step in range(1, args.meta_steps + 1):
        model.train()
        meta_optimizer.zero_grad()

        meta_query_losses = []

        sampled_tasks = train_episodic.sample_task_batch(args.tasks_per_meta_batch)

        # -----------------------------------------------
        # Inner + outer logic
        # For each task:
        # 1. sample support/query
        # 2. compute fast weights from support
        # 3. compute query loss with fast weights
        # 4. accumulate query losses
        # -----------------------------------------------
        for task in sampled_tasks:
            episode = train_episodic.sample_episode(task)

            support_batch = episode["support"]
            query_batch = episode["query"]

            fast_weights = inner_update_fomaml(
                model=model,
                support_batch=support_batch,
                inner_lr=args.inner_lr,
                inner_steps=args.inner_steps,
            )

            query_loss, _ = mse_loss_from_batch(
                model=model,
                batch=query_batch,
                fast_weights=fast_weights,
            )

            meta_query_losses.append(query_loss)

        # -----------------------------------------------
        # Outer loss = mean query loss across tasks
        # This is the true meta-objective in first-order
        # form using task-adapted fast weights.
        # -----------------------------------------------
        meta_loss = torch.stack(meta_query_losses).mean()
        meta_loss.backward()

        torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        meta_optimizer.step()

        if step % args.print_every == 0:
            print(f"[Step {step}/{args.meta_steps}] meta_loss={meta_loss.item():.6f}", flush=True)

        if step % args.val_every == 0:
            val_metrics = evaluate_episodic(
                model=model,
                episodic_data=val_episodic,
                num_episodes=args.val_episodes,
                inner_lr=args.inner_lr,
                inner_steps=args.inner_steps,
            )

            print(
                f"[Validation @ step {step}] "
                f"loss={val_metrics['loss']:.6f} | "
                f"mse_norm={val_metrics['mse_norm']:.6f} | "
                f"mae_norm={val_metrics['mae_norm']:.6f} | "
                f"mean_qwk={val_metrics['mean_qwk']:.6f}",
                flush=True,
            )

            if val_metrics["mse_norm"] < best_val_mse:
                best_val_mse = val_metrics["mse_norm"]
                best_step = step

                best_dir = os.path.join(args.output_dir, "best_meta_model")
                save_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    save_dir=best_dir,
                    extra_info={
                        "best_step": best_step,
                        "best_val_mse_norm": best_val_mse,
                        "best_val_mean_qwk": val_metrics["mean_qwk"],
                        "trainable_mode": args.trainable_mode,
                        "support_size": args.support_size,
                        "query_size": args.query_size,
                        "inner_steps": args.inner_steps,
                        "inner_lr": args.inner_lr,
                        "meta_lr": args.meta_lr,
                        "note": "True first-order MAML style version with fast weights, no deepcopy-based inner loop.",
                    },
                )
                print(f"[Best saved] step={best_step}, val_mse_norm={best_val_mse:.6f}", flush=True)

    print("\nMeta-training complete.", flush=True)
    print(f"Best val mse_norm: {best_val_mse:.6f} at step {best_step}", flush=True)

    # -----------------------------------------------------
    # Final test
    # -----------------------------------------------------
    best_ckpt_path = os.path.join(args.output_dir, "best_meta_model", "meta_model.pt")
    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        model.to(device)

    test_metrics = evaluate_episodic(
        model=model,
        episodic_data=test_episodic,
        num_episodes=args.test_episodes,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
    )

    print(
        f"\n[Test episodic metrics] "
        f"loss={test_metrics['loss']:.6f} | "
        f"mse_norm={test_metrics['mse_norm']:.6f} | "
        f"mae_norm={test_metrics['mae_norm']:.6f} | "
        f"mean_qwk={test_metrics['mean_qwk']:.6f}",
        flush=True,
    )

    with open(os.path.join(args.output_dir, "final_test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()