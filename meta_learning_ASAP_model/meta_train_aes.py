print("Script file started", flush=True)

import os
print("Imported os", flush=True)

import json
print("Imported json", flush=True)

import random
print("Imported random", flush=True)

import argparse
print("Imported argparse", flush=True)

from copy import deepcopy
print("Imported deepcopy", flush=True)

from typing import Dict, List, Tuple
print("Imported typing", flush=True)

import numpy as np
print("Imported numpy", flush=True)

import pandas as pd
print("Imported pandas", flush=True)

import torch
print("Imported torch", flush=True)

import torch.nn as nn
print("Imported torch.nn", flush=True)

import torch.nn.functional as F
print("Imported torch.nn.functional", flush=True)

from transformers import (
    set_seed,
    T5Tokenizer,
    T5ForConditionalGeneration,
)
print("Imported transformers", flush=True)


# =========================================================
# CONSTANTS
# These are reused from your alignment fine-tuning script
# so the prompt descriptions, rubric text, and score ranges
# stay consistent across alignment + meta-training.
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
# ARGUMENT PARSING
# This keeps training settings configurable from CLI so you
# can change support/query size, trainable mode, prompt split,
# learning rates, etc. without editing the script each time.
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Meta-training for AES using a first practical episodic version before true FOMAML."
    )

    # Input data path and aligned checkpoint path
    parser.add_argument("--train_file", type=str, required=True, help="Path to ASAP TSV/CSV file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to alignment finetuned checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")

    # General settings
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_input_len", type=int, default=768)

    # Prompt split used to simulate cross-prompt transfer
    parser.add_argument("--train_prompts", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--val_prompts", type=int, nargs="+", default=[7])
    parser.add_argument("--test_prompts", type=int, nargs="+", default=[8])

    # Optional balancing to avoid some prompt/trait groups dominating training
    parser.add_argument("--max_samples_per_group_train", type=int, default=300)
    parser.add_argument("--max_samples_per_group_val", type=int, default=100)

    # Meta-learning episode settings
    parser.add_argument("--support_size", type=int, default=8)
    parser.add_argument("--query_size", type=int, default=16)
    parser.add_argument("--tasks_per_meta_batch", type=int, default=4)
    parser.add_argument("--meta_steps", type=int, default=2000)

    # Inner-loop and outer-loop settings
    parser.add_argument("--inner_steps", type=int, default=1)
    parser.add_argument("--inner_lr", type=float, default=1e-3)
    parser.add_argument("--meta_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Which parameters to train in this first version
    parser.add_argument("--trainable_mode", type=str, default="head", choices=["head", "last_k", "all"])
    parser.add_argument("--unfreeze_last_k", type=int, default=2)

    # Validation / logging frequency
    parser.add_argument("--val_every", type=int, default=100)
    parser.add_argument("--val_episodes", type=int, default=100)
    parser.add_argument("--test_episodes", type=int, default=200)
    parser.add_argument("--print_every", type=int, default=20)

    return parser.parse_args()


# =========================================================
# SEED SETUP
# Sets all major random sources so sampling and training
# are more reproducible across runs.
# =========================================================
def setup_seed(seed: int):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# DATA LOADING
# Tries CSV first, then TSV fallback, because some ASAP files
# are stored in tab-separated format.
# =========================================================
def load_table(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep="\t")


# =========================================================
# SCORE NORMALIZATION
# Converts raw rubric score into [0,1] normalized range so
# all traits/prompts can be trained on a common scale.
# =========================================================
def normalize_score(score: float, min_score: float, max_score: float) -> float:
    if max_score == min_score:
        return 0.0
    return (score - min_score) / (max_score - min_score)


# =========================================================
# INFER SCORE RANGES
# If a trait/prompt pair is not found in the hardcoded mapping,
# this estimates its min/max score range from the dataset.
# =========================================================
def infer_score_ranges(df: pd.DataFrame, trait_columns: List[str]) -> Dict[Tuple[int, str], Tuple[float, float]]:
    score_ranges = {}

    # Loop through each prompt/essay_set
    for essay_set in sorted(df["essay_set"].dropna().unique()):
        df_set = df[df["essay_set"] == essay_set]

        # For each trait, inspect available scores and infer min/max
        for trait in trait_columns:
            if trait in df_set.columns:
                values = df_set[trait].dropna().astype(float)
                if len(values) > 0:
                    score_ranges[(int(essay_set), trait)] = (float(values.min()), float(values.max()))

    return score_ranges


# =========================================================
# GET SCORE RANGE
# Returns the score range for a given (essay_set, trait) pair.
# Priority:
# 1. predefined SCORE_RANGES
# 2. inferred ranges from the data
# =========================================================
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


# =========================================================
# BUILD INPUT TEXT
# Creates the exact input prompt given to the model.
# This matches your alignment stage style, so meta-learning
# continues from the same prompt-conditioned formulation.
# =========================================================
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


# =========================================================
# WIDE -> LONG CONVERSION
# Converts the original ASAP-style row format:
#   essay | content | organization | ...
# into trait-wise training rows:
#   essay | trait | score_norm | input_text
#
# This is essential because each meta-task is defined as:
#   task = (essay_set, trait)
# =========================================================
def wide_to_long_with_norm(
    df: pd.DataFrame,
    trait_columns: List[str],
    inferred_ranges: Dict[Tuple[int, str], Tuple[float, float]],
) -> pd.DataFrame:
    records = []

    # Loop over essays in the original dataframe
    for _, row in df.iterrows():
        essay_id = row["essay_id"]
        essay_set = int(row["essay_set"])
        essay_text = str(row["essay"])

        # For each essay, create one row per available trait
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


# =========================================================
# BALANCED SUBSET
# Limits the number of examples per (essay_set, trait) group.
# This helps avoid large groups dominating the episodic pool.
# =========================================================
def balanced_subset(df_long: pd.DataFrame, max_per_group: int, seed: int) -> pd.DataFrame:
    parts = []
    grouped = df_long.groupby(["essay_set", "trait"])

    # Process each prompt-trait group separately
    for _, group_df in grouped:
        if len(group_df) > max_per_group:
            group_df = group_df.sample(max_per_group, random_state=seed)
        parts.append(group_df)

    if len(parts) == 0:
        raise ValueError("No data available after grouping.")

    # Shuffle the final concatenated dataframe
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


# =========================================================
# SAMPLE WITHOUT REPLACEMENT
# Used during episodic construction so support/query samples
# do not duplicate within one episode.
# =========================================================
def sample_without_replacement(indices: List[int], n: int) -> List[int]:
    if len(indices) < n:
        raise ValueError(f"Not enough samples. Need {n}, found {len(indices)}")
    return random.sample(indices, n)


# =========================================================
# MODEL
# This first meta-training version uses:
#   T5 encoder + regression head
#
# Why:
# - easier inner-loop optimization
# - more stable than token generation for meta-learning
# - still uses the alignment checkpoint weights
#
# NOTE:
# This is a design change from the alignment stage.
# Alignment used text generation; meta-training here uses
# regression on normalized score.
# =========================================================
class T5EncoderRegressor(nn.Module):
    """
    Uses encoder representations from T5 and predicts a single normalized score.
    """

    def __init__(self, model_path: str):
        super().__init__()

        # Load the previously aligned T5 checkpoint
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_path)
        hidden_size = self.t5.config.d_model

        # Small regression head placed on top of pooled encoder output
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids, attention_mask):
        # Encode input text using the T5 encoder only
        enc_out = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = enc_out.last_hidden_state  # [B, T, H]

        # Mean-pool token embeddings using attention mask
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        # Predict a scalar normalized score
        pred = self.regressor(pooled)

        # Keep output in [0, 1] because targets are normalized
        pred = torch.sigmoid(pred)
        return pred


# =========================================================
# TRAINABLE PARAMETER SELECTION
# Controls which model parts are updated in the outer loop.
#
# Modes:
# - head: only regression head
# - last_k: regression head + last K encoder blocks
# - all: full encoder + head
#
# For first experiments, "head" is safest and cheapest.
# =========================================================
def set_trainable_params(model: T5EncoderRegressor, mode: str, unfreeze_last_k: int):
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    if mode == "head":
        # Train only the regression head
        for p in model.regressor.parameters():
            p.requires_grad = True

    elif mode == "last_k":
        # Train regression head
        for p in model.regressor.parameters():
            p.requires_grad = True

        # Unfreeze the last K encoder blocks
        blocks = model.t5.encoder.block
        k = min(unfreeze_last_k, len(blocks))
        for block in blocks[-k:]:
            for p in block.parameters():
                p.requires_grad = True

        # Also unfreeze final layer norm for encoder
        for p in model.t5.encoder.final_layer_norm.parameters():
            p.requires_grad = True

    elif mode == "all":
        # Train the full encoder and regression head
        for p in model.t5.encoder.parameters():
            p.requires_grad = True
        for p in model.regressor.parameters():
            p.requires_grad = True

    else:
        raise ValueError(f"Unknown mode: {mode}")


# =========================================================
# EPISODIC DATASET
# Groups trait-wise rows into tasks:
#   task = (essay_set, trait)
#
# Each sampled episode contains:
# - support batch
# - query batch
#
# This matches the meta-learning setting where the model
# adapts on support and is evaluated on query.
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

        # Map each task to the row indices belonging to that task
        self.task_to_indices: Dict[Tuple[int, str], List[int]] = {}

        # Build task index mapping
        for idx, row in self.df.iterrows():
            task = (int(row["essay_set"]), str(row["trait"]))
            self.task_to_indices.setdefault(task, []).append(idx)

        # Keep only tasks that have enough examples for one full episode
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
        """
        Samples multiple tasks for one outer meta-batch.
        If there are not enough unique tasks, sampling falls back to replacement.
        """
        if len(self.tasks) >= num_tasks:
            return random.sample(self.tasks, num_tasks)
        return random.choices(self.tasks, k=num_tasks)

    def sample_episode(self, task: Tuple[int, str]) -> Dict[str, torch.Tensor]:
        """
        Samples one support/query split for a given task.
        """
        indices = self.task_to_indices[task]
        chosen = sample_without_replacement(indices, self.support_size + self.query_size)

        # First part becomes support, second part becomes query
        support_indices = chosen[:self.support_size]
        query_indices = chosen[self.support_size:]

        support_df = self.df.iloc[support_indices]
        query_df = self.df.iloc[query_indices]

        return {
            "support": self._encode_batch(support_df),
            "query": self._encode_batch(query_df),
        }

    def _encode_batch(self, batch_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Tokenizes a batch of text prompts and returns tensors plus normalized scores.
        """
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
        }


# =========================================================
# LOSS FUNCTION
# Runs a forward pass and computes MSE between predicted
# normalized score and gold normalized score.
# =========================================================
def mse_loss_from_batch(model: nn.Module, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    preds = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    loss = F.mse_loss(preds, batch["scores"])
    return loss, preds


# =========================================================
# INNER ADAPTATION
# This simulates the inner loop of meta-learning:
# 1. clone the current model
# 2. adapt it on support examples
# 3. return adapted model
#
# IMPORTANT:
# This is a FIRST VERSION / PRACTICAL APPROXIMATION.
# It uses deepcopy(model) and an optimizer on the cloned model.
#
# This is NOT the final true FOMAML implementation.
#
# In the true FOMAML version, this should be replaced with:
# - functional parameter updates
# - no deepcopy(model)
# - explicit fast weights
# - meta-gradients computed from adapted parameters
# =========================================================
def inner_adapt_model(
    model: T5EncoderRegressor,
    support_batch: Dict[str, torch.Tensor],
    inner_lr: float,
    inner_steps: int,
    trainable_mode: str,
    unfreeze_last_k: int,
) -> T5EncoderRegressor:
    """
    Practical first version of inner-loop adaptation.
    """

    # -----------------------------------------------------
    # FIRST-VERSION APPROXIMATION:
    # Clone the model for task-specific adaptation.
    # This is easy to understand and debug, but in the
    # true FOMAML version this should be replaced with
    # functional fast-weight updates instead of deepcopy.
    # -----------------------------------------------------
    adapted_model = deepcopy(model)
    adapted_model.train()

    # Re-apply trainable setting to the adapted model copy
    set_trainable_params(adapted_model, trainable_mode, unfreeze_last_k)

    # -----------------------------------------------------
    # FIRST-VERSION APPROXIMATION:
    # Use a standard optimizer on the copied model.
    # In the true FOMAML script, this should become
    # manual parameter updates:
    #   theta' = theta - alpha * grad
    # without building a separate copied optimizer.
    # -----------------------------------------------------
    inner_optimizer = torch.optim.SGD(
        [p for p in adapted_model.parameters() if p.requires_grad],
        lr=inner_lr,
    )

    # Perform one or more adaptation steps on the support set
    for _ in range(inner_steps):
        inner_optimizer.zero_grad()
        support_loss, _ = mse_loss_from_batch(adapted_model, support_batch)
        support_loss.backward()
        inner_optimizer.step()

    return adapted_model


# =========================================================
# EPISODIC EVALUATION
# For each validation/test episode:
# 1. sample a task
# 2. adapt on support
# 3. evaluate on query
#
# This better reflects few-shot adaptation than plain
# batch evaluation.
#
# IMPORTANT:
# Since this uses the same deepcopy-based adaptation,
# it is also part of the first practical version.
# =========================================================
@torch.no_grad()
def evaluate_episodic(
    model: T5EncoderRegressor,
    episodic_data: EpisodicAESDataset,
    num_episodes: int,
    inner_lr: float,
    inner_steps: int,
    trainable_mode: str,
    unfreeze_last_k: int,
) -> Dict[str, float]:
    all_losses = []
    all_preds = []
    all_targets = []

    # Loop through multiple randomly sampled validation/test episodes
    for _ in range(num_episodes):
        task = random.choice(episodic_data.tasks)
        episode = episodic_data.sample_episode(task)

        # -------------------------------------------------
        # Adapt on support set
        # Gradients must be enabled temporarily because
        # support adaptation needs backward() even inside eval.
        # -------------------------------------------------
        with torch.enable_grad():
            adapted_model = inner_adapt_model(
                model=model,
                support_batch=episode["support"],
                inner_lr=inner_lr,
                inner_steps=inner_steps,
                trainable_mode=trainable_mode,
                unfreeze_last_k=unfreeze_last_k,
            )

        adapted_model.eval()

        # Evaluate the adapted model on the query set
        with torch.no_grad():
            query_loss, query_preds = mse_loss_from_batch(adapted_model, episode["query"])

        all_losses.append(query_loss.item())
        all_preds.extend(query_preds.squeeze(-1).cpu().numpy().tolist())
        all_targets.extend(episode["query"]["scores"].squeeze(-1).cpu().numpy().tolist())

    all_preds = np.array(all_preds, dtype=np.float32)
    all_targets = np.array(all_targets, dtype=np.float32)

    mse = float(np.mean((all_preds - all_targets) ** 2))
    mae = float(np.mean(np.abs(all_preds - all_targets)))

    return {
        "loss": float(np.mean(all_losses)),
        "mse_norm": mse,
        "mae_norm": mae,
    }


# =========================================================
# CHECKPOINT SAVING
# Saves the current best meta-trained model, tokenizer, and
# metadata describing the training setup.
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
# Full training pipeline:
# 1. load raw ASAP-style data
# 2. split by prompts
# 3. convert wide -> long
# 4. balance groups
# 5. build episodic task pools
# 6. meta-train
# 7. validate periodically
# 8. save best model
# 9. test on held-out prompts
# =========================================================
def main():
    print("Entering main function...")

    args = parse_args()
    setup_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Output directory: {args.output_dir}")

    # -----------------------------------------------------
    # Load raw data file
    # -----------------------------------------------------
    df = load_table(args.train_file)

    print(f"Loaded training data with {len(df)} rows.")

    required_cols = {"essay_id", "essay_set", "essay"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Detect which trait columns are actually present in the file
    available_traits = [t for t in TRAIT_COLUMNS if t in df.columns]
    print("Available traits:", available_traits)

    # Infer score ranges if needed
    inferred_score_ranges = infer_score_ranges(df, available_traits)

    # -----------------------------------------------------
    # Prompt-based split:
    # train prompts for meta-training
    # val prompt for episodic validation
    # test prompt for held-out evaluation
    # -----------------------------------------------------
    df_train_pool = df[df["essay_set"].isin(args.train_prompts)].copy()
    df_val_pool = df[df["essay_set"].isin(args.val_prompts)].copy()
    df_test_pool = df[df["essay_set"].isin(args.test_prompts)].copy()

    print("\nPrompt-based split sizes:")
    print("Train pool essays:", len(df_train_pool))
    print("Val pool essays:", len(df_val_pool))
    print("Test pool essays:", len(df_test_pool))

    # -----------------------------------------------------
    # Convert wide-format essay data into trait-wise long format
    # -----------------------------------------------------
    df_train_long = wide_to_long_with_norm(df_train_pool, available_traits, inferred_score_ranges)
    df_val_long = wide_to_long_with_norm(df_val_pool, available_traits, inferred_score_ranges)
    df_test_long = wide_to_long_with_norm(df_test_pool, available_traits, inferred_score_ranges)

    print("\nTrait-wise sizes before balancing:")
    print("Train long:", len(df_train_long))
    print("Val long:", len(df_val_long))
    print("Test long:", len(df_test_long))

    # -----------------------------------------------------
    # Balance train and val groups
    # This reduces dominance of large prompt/trait groups.
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
    df_test_meta = df_test_long.reset_index(drop=True)

    print("\nTrait-wise sizes after balancing:")
    print("Train meta:", len(df_train_meta))
    print("Val meta:", len(df_val_meta))
    print("Test meta:", len(df_test_meta))

    print("\nTrain group counts:")
    print(df_train_meta.groupby(["essay_set", "trait"]).size().head(30))

    print("\nVal group counts:")
    print(df_val_meta.groupby(["essay_set", "trait"]).size().head(30))

    # -----------------------------------------------------
    # Load tokenizer and aligned checkpoint
    # -----------------------------------------------------
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)

    model = T5EncoderRegressor(args.model_path)
    set_trainable_params(model, args.trainable_mode, args.unfreeze_last_k)
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\nTrainable mode: {args.trainable_mode}")
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    # Outer-loop optimizer
    meta_optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.meta_lr,
        weight_decay=args.weight_decay,
    )

    # -----------------------------------------------------
    # Build episodic datasets
    # Each task = (essay_set, trait)
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

    print(f"\nTrain tasks: {len(train_episodic.tasks)}")
    print(f"Val tasks: {len(val_episodic.tasks)}")
    print(f"Test tasks: {len(test_episodic.tasks)}")

    # -----------------------------------------------------
    # Meta-training state tracking
    # -----------------------------------------------------
    best_val_mse = float("inf")
    best_step = -1

    print("\nStarting meta-training...\n")

    # -----------------------------------------------------
    # OUTER META-TRAINING LOOP
    # Each step:
    # 1. sample several tasks
    # 2. adapt a copied model on each task's support set
    # 3. compute query loss for each task
    # 4. aggregate query losses into meta loss
    # 5. update outer model
    # -----------------------------------------------------
    for step in range(1, args.meta_steps + 1):
        model.train()
        meta_optimizer.zero_grad()

        meta_query_losses = []

        # Sample a batch of tasks for this meta-step
        sampled_tasks = train_episodic.sample_task_batch(args.tasks_per_meta_batch)

        # Process each sampled task separately
        for task in sampled_tasks:
            episode = train_episodic.sample_episode(task)

            # -------------------------------------------------
            # FIRST-VERSION APPROXIMATION:
            # Task-specific adaptation via copied model.
            # In the true FOMAML version, this should be
            # replaced by fast-weight updates computed from
            # support gradients directly.
            # -------------------------------------------------
            adapted_model = inner_adapt_model(
                model=model,
                support_batch=episode["support"],
                inner_lr=args.inner_lr,
                inner_steps=args.inner_steps,
                trainable_mode=args.trainable_mode,
                unfreeze_last_k=args.unfreeze_last_k,
            )

            # Evaluate adapted model on query set
            query_loss, _ = mse_loss_from_batch(adapted_model, episode["query"])
            meta_query_losses.append(query_loss)

        # Average query losses across tasks -> outer loss
        meta_loss = torch.stack(meta_query_losses).mean()

        # -------------------------------------------------
        # FIRST-VERSION APPROXIMATION:
        # This outer backward is based on the adapted copied
        # models and is not the clean functional FOMAML form.
        #
        # In the true FOMAML version, the query loss should
        # backprop to the initial parameters through the
        # first-order fast-weight update pipeline.
        # -------------------------------------------------
        meta_loss.backward()

        torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        meta_optimizer.step()

        if step % args.print_every == 0:
            print(f"[Step {step}/{args.meta_steps}] meta_loss={meta_loss.item():.6f}")

        # -------------------------------------------------
        # Periodic episodic validation
        # -------------------------------------------------
        if step % args.val_every == 0:
            val_metrics = evaluate_episodic(
                model=model,
                episodic_data=val_episodic,
                num_episodes=args.val_episodes,
                inner_lr=args.inner_lr,
                inner_steps=args.inner_steps,
                trainable_mode=args.trainable_mode,
                unfreeze_last_k=args.unfreeze_last_k,
            )

            print(
                f"[Validation @ step {step}] "
                f"loss={val_metrics['loss']:.6f} | "
                f"mse_norm={val_metrics['mse_norm']:.6f} | "
                f"mae_norm={val_metrics['mae_norm']:.6f}"
            )

            # Save best model based on validation MSE
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
                        "trainable_mode": args.trainable_mode,
                        "support_size": args.support_size,
                        "query_size": args.query_size,
                        "inner_steps": args.inner_steps,
                        "inner_lr": args.inner_lr,
                        "meta_lr": args.meta_lr,
                        "note": "This checkpoint comes from the first practical episodic version, not yet the true functional FOMAML implementation.",
                    },
                )
                print(f"[Best saved] step={best_step}, val_mse_norm={best_val_mse:.6f}")

    print("\nMeta-training complete.")
    print(f"Best val mse_norm: {best_val_mse:.6f} at step {best_step}")

    # -----------------------------------------------------
    # Reload best checkpoint before final test
    # -----------------------------------------------------
    best_ckpt_path = os.path.join(args.output_dir, "best_meta_model", "meta_model.pt")
    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        model.to(device)

    # -----------------------------------------------------
    # Final episodic test on held-out prompt tasks
    # -----------------------------------------------------
    test_metrics = evaluate_episodic(
        model=model,
        episodic_data=test_episodic,
        num_episodes=args.test_episodes,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        trainable_mode=args.trainable_mode,
        unfreeze_last_k=args.unfreeze_last_k,
    )

    print(
        f"\n[Test episodic metrics] "
        f"loss={test_metrics['loss']:.6f} | "
        f"mse_norm={test_metrics['mse_norm']:.6f} | "
        f"mae_norm={test_metrics['mae_norm']:.6f}"
    )

    with open(os.path.join(args.output_dir, "final_test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()