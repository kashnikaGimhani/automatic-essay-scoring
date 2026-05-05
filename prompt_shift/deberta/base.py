import os
import math
import argparse
from pathlib import Path
import sys
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

PARENT_DIR = str(Path(__file__).resolve().parents[1])
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from utils import (
    TRAIT_COLUMNS,
    ensure_dir,
    save_json,
    set_seed,
    normalize_prompt_id,
    parse_prompt_list,
    build_prompt_text_map,
    build_score_ranges_from_hardcoded,
    build_global_trait_fallback,
    format_metrics_for_print,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Base model training for DeBERTa/Longformer encoder with ordinal/distributional head using prompt-local class indices"
    )

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)

    parser.add_argument("--sep", type=str, default="\t")
    parser.add_argument("--prompt_col", type=str, default="essay_set")
    parser.add_argument("--text_col", type=str, default="essay")

    parser.add_argument("--heldout_prompts", type=str, default="all")
    parser.add_argument("--dev_size", type=float, default=0.1, help="Fraction of source prompts used for source-side dev")

    parser.add_argument("--encoder_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--head_type", type=str, default="distribution", choices=["distribution", "ordinal"])
    parser.add_argument(
        "--num_bins",
        type=int,
        default=0,
        help="Global output bins. If <= 0, infer automatically from the widest prompt/trait range. If smaller than required, it will be increased.",
    )
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean"])
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4, help="Optional larger LR for the head")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=3)

    parser.add_argument("--round_step", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


class OrdinalDistributionalAESModel(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        num_traits: int,
        num_bins: int,
        head_type: str = "distribution",
        dropout: float = 0.1,
        pooling: str = "cls",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.num_traits = num_traits
        self.num_bins = num_bins
        self.head_type = head_type
        self.pooling = pooling

        self.config = AutoConfig.from_pretrained(encoder_name)
        if dropout >= 0:
            for attr in [
                "hidden_dropout_prob",
                "attention_probs_dropout_prob",
                "classifier_dropout",
                "cls_dropout",
                "pooler_dropout",
                "dropout",
            ]:
                if hasattr(self.config, attr):
                    setattr(self.config, attr, dropout)

        self.encoder = AutoModel.from_pretrained(encoder_name, config=self.config)
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        if head_type == "distribution":
            self.head = nn.Linear(hidden_size, num_traits * num_bins)
        elif head_type == "ordinal":
            self.head = nn.Linear(hidden_size, num_traits * (num_bins - 1))
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

    @property
    def model_type(self) -> str:
        return getattr(self.config, "model_type", "")

    def _build_encoder_kwargs(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None and self.model_type not in {"deberta-v2", "longformer", "roberta", "xlm-roberta"}:
            kwargs["token_type_ids"] = token_type_ids
        if self.model_type == "longformer":
            global_attention_mask = torch.zeros_like(attention_mask)
            global_attention_mask[:, 0] = 1
            kwargs["global_attention_mask"] = global_attention_mask
        return kwargs

    def encode(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(**self._build_encoder_kwargs(input_ids, attention_mask, token_type_ids))
        last_hidden = outputs.last_hidden_state
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            return pooled
        return last_hidden[:, 0]

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        pooled = self.encode(input_ids, attention_mask, token_type_ids)
        pooled = self.dropout(pooled)
        logits = self.head(pooled)
        if self.head_type == "distribution":
            return logits.view(-1, self.num_traits, self.num_bins)
        return logits.view(-1, self.num_traits, self.num_bins - 1)

    def predict_class_indices(self, input_ids, attention_mask, valid_class_counts, token_type_ids=None):
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        if self.head_type == "distribution":
            bin_ids = torch.arange(self.num_bins, device=logits.device).view(1, 1, -1)
            valid_bin_mask = bin_ids < valid_class_counts.unsqueeze(-1)
            masked_logits = logits.masked_fill(~valid_bin_mask, -1e9)
            probs = masked_logits.softmax(dim=-1)
            class_idx = (probs * bin_ids.to(probs.dtype)).sum(dim=-1)
            return class_idx, logits

        thr_ids = torch.arange(self.num_bins - 1, device=logits.device).view(1, 1, -1)
        valid_thr_mask = thr_ids < (valid_class_counts - 1).clamp(min=0).unsqueeze(-1)
        probs = torch.sigmoid(logits) * valid_thr_mask.to(logits.dtype)
        class_idx = probs.sum(dim=-1)
        max_idx = (valid_class_counts - 1).clamp(min=0).to(class_idx.dtype)
        class_idx = torch.minimum(class_idx, max_idx)
        return class_idx, logits


class PromptLocalClassIndexDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        trait_cols,
        prompt_col: str,
        text_col: str,
        prompt_text_map,
        score_ranges,
        global_trait_fallback,
        max_length: int,
        round_step: float,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.trait_cols = list(trait_cols)
        self.prompt_col = prompt_col
        self.text_col = text_col
        self.prompt_text_map = prompt_text_map or {}
        self.score_ranges = score_ranges
        self.global_trait_fallback = global_trait_fallback
        self.max_length = max_length
        self.round_step = round_step

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt_id = normalize_prompt_id(row[self.prompt_col])
        prompt_text = self.prompt_text_map.get(prompt_id, f"Prompt {prompt_id}")
        essay_text = str(row.get(self.text_col, ""))
        model_input = f"Prompt: {prompt_text}\n\nEssay:\n{essay_text}"

        encoded = self.tokenizer(
            model_input,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        class_labels = []
        label_mask = []
        valid_class_counts = []
        raw_labels = []

        for trait in self.trait_cols:
            lo, hi = get_score_range(self.score_ranges, prompt_id, trait, self.global_trait_fallback)
            num_classes = get_num_classes(lo, hi, self.round_step)
            valid_class_counts.append(num_classes)

            raw_val = pd.to_numeric(row.get(trait), errors="coerce")
            if pd.isna(raw_val):
                class_labels.append(0.0)
                raw_labels.append(float("nan"))
                label_mask.append(0.0)
            else:
                class_idx = score_to_class_index(float(raw_val), lo, hi, self.round_step)
                class_labels.append(float(class_idx))
                raw_labels.append(float(raw_val))
                label_mask.append(1.0)

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(class_labels, dtype=torch.float32),
            "label_mask": torch.tensor(label_mask, dtype=torch.float32),
            "valid_class_counts": torch.tensor(valid_class_counts, dtype=torch.long),
            "raw_labels": torch.tensor(raw_labels, dtype=torch.float32),
        }
        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"].squeeze(0)
        return item


def should_use_fast_tokenizer(model_name: str) -> bool:
    name = model_name.lower()
    return not ("deberta-v2" in name or "deberta-v3" in name)


def get_score_range(score_ranges, prompt_id: str, trait: str, global_trait_fallback):
    prompt_id = normalize_prompt_id(prompt_id)

    def parse_range_obj(r):
        if isinstance(r, dict):
            if "min" in r and "max" in r:
                return float(r["min"]), float(r["max"])
            if "low" in r and "high" in r:
                return float(r["low"]), float(r["high"])
            raise ValueError(f"Unsupported score range dict format: {r}")

        if isinstance(r, (list, tuple)) and len(r) >= 2:
            return float(r[0]), float(r[1])

        raise ValueError(f"Unsupported score range format: {r}")

    if isinstance(score_ranges, dict):
        if (prompt_id, trait) in score_ranges:
            return parse_range_obj(score_ranges[(prompt_id, trait)])

        if (
            prompt_id in score_ranges
            and isinstance(score_ranges[prompt_id], dict)
            and trait in score_ranges[prompt_id]
        ):
            return parse_range_obj(score_ranges[prompt_id][trait])

    fallback_obj = global_trait_fallback.get(trait, (0.0, 1.0))
    return parse_range_obj(fallback_obj)


def get_num_classes(lo: float, hi: float, round_step: float = 1.0) -> int:
    if round_step <= 0:
        raise ValueError("round_step must be > 0")
    width = (float(hi) - float(lo)) / float(round_step)
    return int(round(width)) + 1


def score_to_class_index(raw_score: float, lo: float, hi: float, round_step: float = 1.0) -> int:
    clipped = min(max(float(raw_score), float(lo)), float(hi))
    idx = int(round((clipped - float(lo)) / float(round_step)))
    return max(0, min(idx, get_num_classes(lo, hi, round_step) - 1))


def class_index_to_score(class_idx: float, lo: float, hi: float, round_step: float = 1.0) -> float:
    max_idx = get_num_classes(lo, hi, round_step) - 1
    idx = min(max(float(class_idx), 0.0), float(max_idx))
    value = float(lo) + idx * float(round_step)
    value = round(value / round_step) * round_step
    return min(max(value, float(lo)), float(hi))


def infer_required_num_bins(df: pd.DataFrame, prompt_col: str, trait_cols, score_ranges, global_trait_fallback, round_step: float) -> int:
    max_classes = 1
    prompts = df[prompt_col].astype(str).map(normalize_prompt_id).unique().tolist()
    for prompt_id in prompts:
        for trait in trait_cols:
            lo, hi = get_score_range(score_ranges, prompt_id, trait, global_trait_fallback)
            max_classes = max(max_classes, get_num_classes(lo, hi, round_step))
    return max_classes


def build_distribution_targets(class_labels: torch.Tensor, num_bins: int) -> torch.Tensor:
    class_idx = class_labels.long().clamp(0, num_bins - 1)
    target = torch.zeros(*class_labels.shape, num_bins, device=class_labels.device, dtype=torch.float32)
    target.scatter_(-1, class_idx.unsqueeze(-1), 1.0)
    return target


def build_ordinal_targets(class_labels: torch.Tensor, num_bins: int) -> torch.Tensor:
    class_idx = class_labels.long().clamp(0, num_bins - 1)
    thresholds = torch.arange(num_bins - 1, device=class_labels.device).view(1, 1, -1)
    return (class_idx.unsqueeze(-1) > thresholds).float()


def masked_head_loss(logits, labels, label_mask, valid_class_counts, head_type, num_bins):
    if head_type == "distribution":
        logits_fp32 = logits.float()

        class_ids = labels.long().clamp(min=0)
        total_loss = logits_fp32.new_tensor(0.0)
        active_count = 0

        for t in range(logits_fp32.size(1)):
            active = label_mask[:, t] > 0
            if not active.any():
                continue

            counts_t = valid_class_counts[:, t].long().clamp(min=1, max=num_bins)
            valid_bin_mask = (
                torch.arange(num_bins, device=logits_fp32.device)[None, :]
                < counts_t[:, None]
            )  # [B, num_bins]

            masked_logits = logits_fp32[:, t, :].masked_fill(~valid_bin_mask, -1e9)
            target_t = class_ids[:, t].clamp(min=0, max=num_bins - 1)

            loss_t = torch.nn.functional.cross_entropy(
                masked_logits[active],
                target_t[active],
                reduction="mean",
            )
            total_loss = total_loss + loss_t
            active_count += 1

        if active_count == 0:
            return logits_fp32.new_tensor(0.0)

        return total_loss / active_count


def safe_rmse(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return float("nan")
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def safe_qwk(y_true: List[float], y_pred: List[float]) -> float:
    if len(y_true) < 2:
        return float("nan")
    try:
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except Exception:
        return float("nan")


def get_dataset_df(dataset):
    for attr in ["df", "dataframe", "data"]:
        if hasattr(dataset, attr):
            df = getattr(dataset, attr)
            if isinstance(df, pd.DataFrame):
                return df.reset_index(drop=True)
    raise AttributeError("Could not find underlying dataframe on dataset. Expected dataset.df or dataset.dataframe.")


@torch.no_grad()
def evaluate(model, dataloader, dataset, trait_cols, score_ranges, global_trait_fallback, device, round_step):
    model.eval()

    df = get_dataset_df(dataset)
    prompt_col = getattr(dataset, "prompt_col", "essay_set")

    total_loss = 0.0
    total_batches = 0
    all_pred_class = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        label_mask = batch["label_mask"].to(device)
        valid_class_counts = batch["valid_class_counts"].to(device)
        token_type_ids = batch["token_type_ids"].to(device) if "token_type_ids" in batch else None

        pred_class, logits = model.predict_class_indices(
            input_ids=input_ids,
            attention_mask=attention_mask,
            valid_class_counts=valid_class_counts,
            token_type_ids=token_type_ids,
        )
        loss = masked_head_loss(logits, labels, label_mask, valid_class_counts, model.head_type, model.num_bins)
        total_loss += float(loss.item())
        total_batches += 1
        all_pred_class.append(pred_class.detach().cpu())

    pred_class = torch.cat(all_pred_class, dim=0).numpy() if all_pred_class else np.zeros((0, len(trait_cols)))

    trait_metrics = {}
    all_true_flat = []
    all_pred_flat = []

    for t_idx, trait in enumerate(trait_cols):
        y_true = []
        y_pred = []
        for row_idx in range(len(df)):
            true_val = pd.to_numeric(df.iloc[row_idx].get(trait), errors="coerce")
            if pd.isna(true_val):
                continue
            prompt_id = df.iloc[row_idx][prompt_col]
            lo, hi = get_score_range(score_ranges, prompt_id, trait, global_trait_fallback)
            pred_score = class_index_to_score(pred_class[row_idx, t_idx], lo, hi, round_step)
            y_true.append(float(true_val))
            y_pred.append(float(pred_score))
            all_true_flat.append(float(true_val))
            all_pred_flat.append(float(pred_score))

        trait_metrics[trait] = {
            "n": len(y_true),
            "qwk": safe_qwk(y_true, y_pred),
            "rmse": safe_rmse(y_true, y_pred),
        }

    mean_qwk = float(np.nanmean([m["qwk"] for m in trait_metrics.values()])) if trait_metrics else float("nan")
    mean_rmse = float(np.nanmean([m["rmse"] for m in trait_metrics.values()])) if trait_metrics else float("nan")

    return {
        "loss": total_loss / max(total_batches, 1),
        "mean_qwk": mean_qwk,
        "mean_rmse": mean_rmse,
        "trait_metrics": trait_metrics,
        "overall_flat_qwk": safe_qwk(all_true_flat, all_pred_flat),
        "overall_flat_rmse": safe_rmse(all_true_flat, all_pred_flat),
    }


def build_param_groups(model, encoder_lr: float, head_lr: float, weight_decay: float):
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

    encoder_named = list(model.encoder.named_parameters())
    head_named = list(model.head.named_parameters())

    groups = [
        {
            "params": [p for n, p in encoder_named if p.requires_grad and not any(nd in n for nd in no_decay)],
            "lr": encoder_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in encoder_named if p.requires_grad and any(nd in n for nd in no_decay)],
            "lr": encoder_lr,
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in head_named if p.requires_grad],
            "lr": head_lr,
            "weight_decay": weight_decay,
        },
    ]
    return [g for g in groups if g["params"]]


def split_source_train_dev(source_df: pd.DataFrame, prompt_col: str, dev_size: float, seed: int):
    if dev_size <= 0.0:
        return source_df.reset_index(drop=True), source_df.iloc[:0].copy().reset_index(drop=True)

    stratify = None
    vc = source_df[prompt_col].astype(str).value_counts()
    if len(vc) > 1 and (vc >= 2).all():
        stratify = source_df[prompt_col].astype(str)

    train_df, dev_df = train_test_split(
        source_df,
        test_size=dev_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), dev_df.reset_index(drop=True)


def train_one_base(
    model,
    train_loader,
    dev_loader,
    dev_dataset,
    heldout_loader,
    heldout_dataset,
    trait_cols,
    score_ranges,
    global_trait_fallback,
    device,
    args,
    save_dir,
):
    optimizer = torch.optim.AdamW(
        build_param_groups(model, args.lr, args.head_lr, args.weight_decay),
    )

    total_update_steps = max(1, math.ceil(len(train_loader) / args.grad_accum_steps) * args.num_epochs)
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_dev_qwk = -1e9
    best_epoch = -1
    early_stop_counter = 0
    best_ckpt_path = os.path.join(save_dir, "best_model.pt")
    history = []

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        num_steps = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}", leave=False)
        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            label_mask = batch["label_mask"].to(device)
            valid_class_counts = batch["valid_class_counts"].to(device)
            token_type_ids = batch["token_type_ids"].to(device) if "token_type_ids" in batch else None

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                loss = masked_head_loss(logits, labels, label_mask, valid_class_counts, model.head_type, model.num_bins)
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
        dev_metrics = evaluate(
            model=model,
            dataloader=dev_loader,
            dataset=dev_dataset,
            trait_cols=trait_cols,
            score_ranges=score_ranges,
            global_trait_fallback=global_trait_fallback,
            device=device,
            round_step=args.round_step,
        )
        heldout_metrics = evaluate(
            model=model,
            dataloader=heldout_loader,
            dataset=heldout_dataset,
            trait_cols=trait_cols,
            score_ranges=score_ranges,
            global_trait_fallback=global_trait_fallback,
            device=device,
            round_step=args.round_step,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "source_dev_loss": dev_metrics["loss"],
                "source_dev_mean_qwk": dev_metrics["mean_qwk"],
                "source_dev_mean_rmse": dev_metrics["mean_rmse"],
                "heldout_zero_shot_mean_qwk": heldout_metrics["mean_qwk"],
                "heldout_zero_shot_mean_rmse": heldout_metrics["mean_rmse"],
            }
        )

        improved = not math.isnan(dev_metrics["mean_qwk"]) and dev_metrics["mean_qwk"] > best_dev_qwk
        if improved:
            best_dev_qwk = dev_metrics["mean_qwk"]
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_dev_mean_qwk": best_dev_qwk,
                    "trait_cols": trait_cols,
                    "encoder_name": model.encoder_name,
                    "head_type": model.head_type,
                    "num_bins": model.num_bins,
                    "pooling": model.pooling,
                    "dropout": args.dropout,
                    "round_step": args.round_step,
                    "target_encoding": "prompt_local_class_index",
                },
                best_ckpt_path,
            )
        else:
            early_stop_counter += 1

        print(format_metrics_for_print(f"Source dev epoch {epoch}", dev_metrics))
        print(format_metrics_for_print(f"Heldout zero-shot epoch {epoch}", heldout_metrics))

        if early_stop_counter >= args.patience:
            break

    if os.path.exists(best_ckpt_path):
        best_state = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_state["model_state_dict"])

    best_source_dev = evaluate(
        model=model,
        dataloader=dev_loader,
        dataset=dev_dataset,
        trait_cols=trait_cols,
        score_ranges=score_ranges,
        global_trait_fallback=global_trait_fallback,
        device=device,
        round_step=args.round_step,
    )
    best_heldout_zero = evaluate(
        model=model,
        dataloader=heldout_loader,
        dataset=heldout_dataset,
        trait_cols=trait_cols,
        score_ranges=score_ranges,
        global_trait_fallback=global_trait_fallback,
        device=device,
        round_step=args.round_step,
    )

    return model, history, best_epoch, best_dev_qwk, best_source_dev, best_heldout_zero


def main():
    args = parse_args()
    ensure_dir(args.output_root)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_df = pd.read_csv(args.data_path, sep=args.sep)
    full_df[args.prompt_col] = full_df[args.prompt_col].apply(normalize_prompt_id)
    for trait in TRAIT_COLUMNS:
        if trait in full_df.columns:
            full_df[trait] = pd.to_numeric(full_df[trait], errors="coerce")

    all_prompts = sorted(full_df[args.prompt_col].astype(str).unique().tolist())
    heldout_prompts = parse_prompt_list(args.heldout_prompts, all_prompts)

    prompt_text_map = build_prompt_text_map()
    score_ranges = build_score_ranges_from_hardcoded()
    global_trait_fallback = build_global_trait_fallback(full_df, TRAIT_COLUMNS)

    required_num_bins = infer_required_num_bins(
        df=full_df,
        prompt_col=args.prompt_col,
        trait_cols=TRAIT_COLUMNS,
        score_ranges=score_ranges,
        global_trait_fallback=global_trait_fallback,
        round_step=args.round_step,
    )
    effective_num_bins = required_num_bins if args.num_bins <= 0 else max(args.num_bins, required_num_bins)

    print(
        f"Encoder: {args.encoder_name} | head_type: {args.head_type} | "
        f"required_num_bins: {required_num_bins} | using_num_bins: {effective_num_bins}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.encoder_name,
        use_fast=should_use_fast_tokenizer(args.encoder_name),
    )

    summary_rows = []

    for heldout_prompt in heldout_prompts:
        print(f"\n=== Training base model with heldout prompt {heldout_prompt} ===")
        heldout_df = full_df[full_df[args.prompt_col].astype(str) == str(heldout_prompt)].copy().reset_index(drop=True)
        source_df = full_df[full_df[args.prompt_col].astype(str) != str(heldout_prompt)].copy().reset_index(drop=True)

        if len(heldout_df) == 0 or len(source_df) == 0:
            print(f"Skipping heldout={heldout_prompt}: insufficient data")
            continue

        source_train_df, source_dev_df = split_source_train_dev(source_df, args.prompt_col, args.dev_size, args.seed)

        train_dataset = PromptLocalClassIndexDataset(
            df=source_train_df,
            tokenizer=tokenizer,
            trait_cols=TRAIT_COLUMNS,
            prompt_col=args.prompt_col,
            text_col=args.text_col,
            prompt_text_map=prompt_text_map,
            score_ranges=score_ranges,
            global_trait_fallback=global_trait_fallback,
            max_length=args.max_length,
            round_step=args.round_step,
        )
        dev_dataset = PromptLocalClassIndexDataset(
            df=source_dev_df,
            tokenizer=tokenizer,
            trait_cols=TRAIT_COLUMNS,
            prompt_col=args.prompt_col,
            text_col=args.text_col,
            prompt_text_map=prompt_text_map,
            score_ranges=score_ranges,
            global_trait_fallback=global_trait_fallback,
            max_length=args.max_length,
            round_step=args.round_step,
        )
        heldout_dataset = PromptLocalClassIndexDataset(
            df=heldout_df,
            tokenizer=tokenizer,
            trait_cols=TRAIT_COLUMNS,
            prompt_col=args.prompt_col,
            text_col=args.text_col,
            prompt_text_map=prompt_text_map,
            score_ranges=score_ranges,
            global_trait_fallback=global_trait_fallback,
            max_length=args.max_length,
            round_step=args.round_step,
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        heldout_loader = DataLoader(heldout_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        model = OrdinalDistributionalAESModel(
            encoder_name=args.encoder_name,
            num_traits=len(TRAIT_COLUMNS),
            num_bins=effective_num_bins,
            head_type=args.head_type,
            dropout=args.dropout,
            pooling=args.pooling,
            gradient_checkpointing=args.gradient_checkpointing,
        ).to(device)

        save_dir = os.path.join(args.output_root, f"base_prompt{heldout_prompt}", "best_checkpoint")
        ensure_dir(save_dir)

        model, history, best_epoch, best_dev_qwk, final_source_dev, final_heldout_zero = train_one_base(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            dev_dataset=dev_dataset,
            heldout_loader=heldout_loader,
            heldout_dataset=heldout_dataset,
            trait_cols=TRAIT_COLUMNS,
            score_ranges=score_ranges,
            global_trait_fallback=global_trait_fallback,
            device=device,
            args=args,
            save_dir=save_dir,
        )

        tokenizer.save_pretrained(save_dir)
        model.config.save_pretrained(save_dir)
        save_json({"history": history}, os.path.join(save_dir, "training_history.json"))
        save_json(final_source_dev, os.path.join(save_dir, "source_dev_metrics.json"))
        save_json(final_heldout_zero, os.path.join(save_dir, "heldout_zero_shot_metrics.json"))
        save_json(
            {
                "heldout_prompt": heldout_prompt,
                "encoder_name": args.encoder_name,
                "head_type": args.head_type,
                "num_bins": effective_num_bins,
                "required_num_bins": required_num_bins,
                "target_encoding": "prompt_local_class_index",
                "pooling": args.pooling,
                "max_length": args.max_length,
                "batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "num_epochs": args.num_epochs,
                "lr": args.lr,
                "head_lr": args.head_lr,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "grad_accum_steps": args.grad_accum_steps,
                "dropout": args.dropout,
                "round_step": args.round_step,
                "source_train_n": len(source_train_df),
                "source_dev_n": len(source_dev_df),
                "heldout_n": len(heldout_df),
                "best_epoch": best_epoch,
                "best_source_dev_mean_qwk": best_dev_qwk,
            },
            os.path.join(save_dir, "run_config.json"),
        )

        print(format_metrics_for_print("Best source dev", final_source_dev))
        print(format_metrics_for_print("Best heldout zero-shot", final_heldout_zero))

        summary_rows.append(
            {
                "heldout_prompt": heldout_prompt,
                "encoder_name": args.encoder_name,
                "head_type": args.head_type,
                "num_bins": effective_num_bins,
                "required_num_bins": required_num_bins,
                "target_encoding": "prompt_local_class_index",
                "pooling": args.pooling,
                "source_train_n": len(source_train_df),
                "source_dev_n": len(source_dev_df),
                "heldout_n": len(heldout_df),
                "best_epoch": best_epoch,
                "best_source_dev_mean_qwk": best_dev_qwk,
                "best_source_dev_mean_rmse": final_source_dev["mean_rmse"],
                "best_heldout_zero_mean_qwk": final_heldout_zero["mean_qwk"],
                "best_heldout_zero_mean_rmse": final_heldout_zero["mean_rmse"],
            }
        )

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(os.path.join(args.output_root, "base_summary.csv"), index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
