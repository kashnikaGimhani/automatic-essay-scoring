import os
import random
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import (
    set_seed,
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


# =========================================================
# CONSTANTS
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
    parser = argparse.ArgumentParser(description="Lightweight alignment finetuning for trait-wise ASAP scoring with T5.")

    parser.add_argument("--train_file", type=str, required=True, help="Path to ASAP TSV/CSV file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained T5 checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_input_len", type=int, default=768)
    parser.add_argument("--max_target_len", type=int, default=8)

    parser.add_argument("--max_samples_per_group_train", type=int, default=100)
    parser.add_argument("--max_samples_per_group_val", type=int, default=100)

    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--grad_accum", type=int, default=4)

    parser.add_argument("--train_prompts", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--val_prompts", type=int, nargs="+", default=[7])
    parser.add_argument("--test_prompts", type=int, nargs="+", default=[8])

    parser.add_argument("--fp16", action="store_true", help="Enable fp16 training.")
    parser.add_argument("--no_fp16", action="store_true", help="Disable fp16 training.")
    parser.add_argument("--logging_steps", type=int, default=50)

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


def format_target(score_norm: float) -> str:
    return f"{score_norm:.3f}"


def infer_score_ranges(df: pd.DataFrame, trait_columns: List[str]) -> Dict[Tuple[int, str], Tuple[float, float]]:
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
                        "target_text": format_target(score_norm),
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


def safe_parse_float(text: str) -> float:
    try:
        return float(text.strip())
    except Exception:
        return 0.0


def compute_metrics_builder(tokenizer: T5Tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

        y_pred = np.array([safe_parse_float(x) for x in pred_texts], dtype=np.float32)
        y_true = np.array([safe_parse_float(x) for x in label_texts], dtype=np.float32)

        mse = float(np.mean((y_pred - y_true) ** 2))
        mae = float(np.mean(np.abs(y_pred - y_true)))

        return {
            "mse_norm": mse,
            "mae_norm": mae,
        }

    return compute_metrics


def predict_score_text(
    model,
    tokenizer,
    text: str,
    max_input_len: int = 768,
    max_target_len: int = 8,
) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_length=max_target_len,
            num_beams=1,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


# =========================================================
# DATASET
# =========================================================

class TraitwiseT5Dataset(Dataset):
    def __init__(self, df_data: pd.DataFrame, tokenizer: T5Tokenizer, max_input_len: int, max_target_len: int):
        self.df = df_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        model_inputs = self.tokenizer(
            row["input_text"],
            max_length=self.max_input_len,
            truncation=True,
            padding=False,
        )

        labels = self.tokenizer(
            text_target=row["target_text"],
            max_length=self.max_target_len,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


# =========================================================
# MAIN
# =========================================================

def main():
    args = parse_args()

    # fp16 = False if args.no_fp16 else (args.fp16 or torch.cuda.is_available())
    fp16 = False
    setup_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_table(args.train_file)

    required_cols = {"essay_id", "essay_set", "essay"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    available_traits = [t for t in TRAIT_COLUMNS if t in df.columns]
    print("Available traits:", available_traits)

    inferred_score_ranges = infer_score_ranges(df, available_traits)

    print("\nSample inferred score ranges:")
    for k, v in list(inferred_score_ranges.items())[:10]:
        print(k, "->", v)

    df_train_pool = df[df["essay_set"].isin(args.train_prompts)].copy()
    df_val_pool = df[df["essay_set"].isin(args.val_prompts)].copy()
    df_test_pool = df[df["essay_set"].isin(args.test_prompts)].copy()

    print("\nPrompt-based split sizes:")
    print("Train pool essays:", len(df_train_pool))
    print("Val pool essays:", len(df_val_pool))
    print("Test pool essays:", len(df_test_pool))

    df_train_long = wide_to_long_with_norm(df_train_pool, available_traits, inferred_score_ranges)
    df_val_long = wide_to_long_with_norm(df_val_pool, available_traits, inferred_score_ranges)
    df_test_long = wide_to_long_with_norm(df_test_pool, available_traits, inferred_score_ranges)

    print("\nTrait-wise sizes before balancing:")
    print("Train long:", len(df_train_long))
    print("Val long:", len(df_val_long))
    print("Test long:", len(df_test_long))

    df_train_align = balanced_subset(
        df_train_long,
        max_per_group=args.max_samples_per_group_train,
        seed=args.seed,
    )
    df_val_align = balanced_subset(
        df_val_long,
        max_per_group=args.max_samples_per_group_val,
        seed=args.seed,
    )
    df_test_eval = df_test_long.reset_index(drop=True)

    print("\nTrait-wise sizes after balancing:")
    print("Train align:", len(df_train_align))
    print("Val align:", len(df_val_align))
    print("Test eval:", len(df_test_eval))

    print("\nTrain group counts:")
    print(df_train_align.groupby(["essay_set", "trait"]).size().head(20))

    print("\nVal group counts:")
    print(df_val_align.groupby(["essay_set", "trait"]).size().head(20))

    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)

    train_dataset = TraitwiseT5Dataset(
        df_train_align, tokenizer, args.max_input_len, args.max_target_len
    )
    val_dataset = TraitwiseT5Dataset(
        df_val_align, tokenizer, args.max_input_len, args.max_target_len
    )
    test_dataset = TraitwiseT5Dataset(
        df_test_eval, tokenizer, args.max_input_len, args.max_target_len
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        logging_dir=f"{args.output_dir}/logs",
        #for tensorboard
        report_to="tensorboard",       
        logging_strategy="steps", 
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="mse_norm",
        greater_is_better=False,
        fp16=fp16,
        predict_with_generate=True,
        generation_max_length=args.max_target_len,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(tokenizer),
    )

    trainer.train()

    best_model_dir = os.path.join(args.output_dir, "best_alignment_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    print(f"\nSaved best model to: {best_model_dir}")

    val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    print("\nValidation metrics:")
    print(val_metrics)

    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print("\nTest metrics on held-out prompt(s):")
    print(test_metrics)

    print("\nSample predictions on validation set:")

    for i in range(5):
        row = df_val_align.iloc[i]

        pred = predict_score_text(
            model,
            tokenizer,
            row["input_text"],
            max_input_len=args.max_input_len,
            max_target_len=args.max_target_len
        )

        print("Trait:", row["trait"])
        print("Target:", row["target_text"])
        print("Pred:", pred)
        print("-" * 40)



if __name__ == "__main__":
    main()