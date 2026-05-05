import os
import math
import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from torch.utils.data import DataLoader
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
    parse_int_list,
    build_prompt_text_map,
    build_score_ranges_from_hardcoded,
    build_global_trait_fallback,
    AESDataset,
    format_metrics_for_print,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Head-only adaptation with DeBERTa/Longformer encoder and ordinal/distributional head"
    )

    parser.add_argument("--data_path", type=str, required=True, help="Full original dataset; used only for global fallback ranges")
    parser.add_argument("--split_root", type=str, required=True, help="Root directory created by create_target_fewshot_splits.py")
    parser.add_argument("--base_root", type=str, required=True, help="Root containing base checkpoints, e.g. base_root/base_prompt2/best_checkpoint")
    parser.add_argument("--output_root", type=str, required=True)

    parser.add_argument("--sep", type=str, default="\t")
    parser.add_argument("--prompt_col", type=str, default="essay_set")
    parser.add_argument("--text_col", type=str, default="essay")

    parser.add_argument("--heldout_prompts", type=str, default="all")
    parser.add_argument("--fewshot_sizes", type=str, default="8,16,32,64,128")

    parser.add_argument("--encoder_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--head_type", type=str, default="distribution", choices=["distribution", "ordinal"])
    parser.add_argument("--num_bins", type=int, default=6, help="Shared normalized bins used by ordinal/distributional head")
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean"])
    parser.add_argument("--use_first_token_global_attention", action="store_true", help="Useful for Longformer models")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)

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

    def predict_normalized(self, input_ids, attention_mask, token_type_ids=None):
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        if self.head_type == "distribution":
            probs = logits.softmax(dim=-1)
            bins = torch.arange(self.num_bins, device=probs.device, dtype=probs.dtype)
            indices = (probs * bins.view(1, 1, -1)).sum(dim=-1)
            return indices / max(self.num_bins - 1, 1), logits
        probs = torch.sigmoid(logits)
        indices = probs.sum(dim=-1)
        return indices / max(self.num_bins - 1, 1), logits



def freeze_encoder_only(model: OrdinalDistributionalAESModel):
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
    if hasattr(model, "dropout"):
        for p in model.dropout.parameters():
            p.requires_grad = True



def build_soft_distribution_targets(labels: torch.Tensor, num_bins: int) -> torch.Tensor:
    labels = labels.clamp(0.0, 1.0)
    positions = labels * (num_bins - 1)
    lower = torch.floor(positions).long()
    upper = torch.ceil(positions).long()
    upper = upper.clamp(0, num_bins - 1)
    lower = lower.clamp(0, num_bins - 1)
    upper_w = positions - lower.float()
    lower_w = 1.0 - upper_w

    target = torch.zeros(*labels.shape, num_bins, device=labels.device, dtype=labels.dtype)
    target.scatter_add_(-1, lower.unsqueeze(-1), lower_w.unsqueeze(-1))
    target.scatter_add_(-1, upper.unsqueeze(-1), upper_w.unsqueeze(-1))
    return target



def build_ordinal_targets(labels: torch.Tensor, num_bins: int) -> torch.Tensor:
    class_idx = torch.round(labels.clamp(0.0, 1.0) * (num_bins - 1)).long()
    thresholds = torch.arange(num_bins - 1, device=labels.device).view(1, 1, -1)
    return (class_idx.unsqueeze(-1) > thresholds).float()



def masked_head_loss(logits: torch.Tensor, labels: torch.Tensor, label_mask: torch.Tensor, head_type: str, num_bins: int) -> torch.Tensor:
    if head_type == "distribution":
        soft_targets = build_soft_distribution_targets(labels, num_bins)
        log_probs = F.log_softmax(logits, dim=-1)
        per_trait = -(soft_targets * log_probs).sum(dim=-1)
        masked = per_trait * label_mask.float()
        denom = label_mask.float().sum().clamp(min=1.0)
        return masked.sum() / denom

    ordinal_targets = build_ordinal_targets(labels, num_bins)
    bce = F.binary_cross_entropy_with_logits(logits, ordinal_targets, reduction="none").mean(dim=-1)
    masked = bce * label_mask.float()
    denom = label_mask.float().sum().clamp(min=1.0)
    return masked.sum() / denom



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

        if prompt_id in score_ranges and isinstance(score_ranges[prompt_id], dict) and trait in score_ranges[prompt_id]:
            return parse_range_obj(score_ranges[prompt_id][trait])

    lo, hi = global_trait_fallback.get(trait, (0.0, 1.0))
    return float(lo), float(hi)



def denormalize_score(norm_value: float, lo: float, hi: float, round_step: float = 1.0) -> float:
    value = lo + float(norm_value) * (hi - lo)
    value = min(max(value, lo), hi)
    if round_step and round_step > 0:
        value = round(value / round_step) * round_step
    value = min(max(value, lo), hi)
    return value



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
def evaluate(
    model,
    dataloader,
    dataset,
    trait_cols,
    score_ranges,
    global_trait_fallback,
    device,
    round_step,
):
    model.eval()

    df = get_dataset_df(dataset)
    prompt_col = getattr(dataset, "prompt_col", "essay_set")

    total_loss = 0.0
    total_batches = 0
    all_pred_norm = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        label_mask = batch["label_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device) if "token_type_ids" in batch else None

        pred_norm, logits = model.predict_normalized(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        loss = masked_head_loss(logits, labels, label_mask, model.head_type, model.num_bins)
        total_loss += float(loss.item())
        total_batches += 1
        all_pred_norm.append(pred_norm.detach().cpu())

    pred_norm = torch.cat(all_pred_norm, dim=0).numpy() if all_pred_norm else np.zeros((0, len(trait_cols)))

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
            pred_score = denormalize_score(pred_norm[row_idx, t_idx], lo, hi, round_step)
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



def load_base_model(base_ckpt_dir: str, device: torch.device, args):
    ckpt_path = os.path.join(base_ckpt_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Base checkpoint not found: {ckpt_path}. You need base checkpoints trained with the same encoder/head family."
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    trait_cols = ckpt.get("trait_cols", TRAIT_COLUMNS)

    encoder_name = ckpt.get("encoder_name", args.encoder_name)
    head_type = ckpt.get("head_type", args.head_type)
    num_bins = ckpt.get("num_bins", args.num_bins)
    pooling = ckpt.get("pooling", args.pooling)
    dropout = ckpt.get("dropout", args.dropout)

    model = OrdinalDistributionalAESModel(
        encoder_name=encoder_name,
        num_traits=len(trait_cols),
        num_bins=num_bins,
        head_type=head_type,
        dropout=dropout,
        pooling=pooling,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    try:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            "Base checkpoint is incompatible with this architecture. "
            "Train/export a new base model checkpoint using the same encoder_name/head_type/num_bins.\n"
            f"Original load error: {e}"
        )

    model.to(device)
    return model, trait_cols, ckpt



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
    freeze_encoder_only(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
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
    best_ckpt_path = os.path.join(run_dir, "best_head_only.pt")

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
            token_type_ids = batch["token_type_ids"].to(device) if "token_type_ids" in batch else None

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                loss = masked_head_loss(
                    logits=logits,
                    labels=labels,
                    label_mask=label_mask,
                    head_type=model.head_type,
                    num_bins=model.num_bins,
                )
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
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

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_loss": dev_metrics["loss"],
                "dev_mean_qwk": dev_metrics["mean_qwk"],
                "dev_mean_rmse": dev_metrics["mean_rmse"],
            }
        )

        dev_qwk = dev_metrics["mean_qwk"]
        improved = not math.isnan(dev_qwk) and dev_qwk > best_dev_qwk

        if improved:
            best_dev_qwk = dev_qwk
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
                },
                best_ckpt_path,
            )
        else:
            early_stop_counter += 1

        if early_stop_counter >= args.patience:
            break

    if os.path.exists(best_ckpt_path):
        best_state = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_state["model_state_dict"])

    return model, history, best_epoch, best_dev_qwk



def main():
    args = parse_args()
    ensure_dir(args.output_root)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Encoder: {args.encoder_name} | head_type: {args.head_type} | num_bins: {args.num_bins}")

    full_df = pd.read_csv(args.data_path, sep=args.sep)
    full_df[args.prompt_col] = full_df[args.prompt_col].apply(normalize_prompt_id)

    for trait in TRAIT_COLUMNS:
        if trait in full_df.columns:
            full_df[trait] = pd.to_numeric(full_df[trait], errors="coerce")

    all_prompts = sorted(full_df[args.prompt_col].astype(str).unique().tolist())
    heldout_prompts = parse_prompt_list(args.heldout_prompts, all_prompts)
    fewshot_sizes = sorted(parse_int_list(args.fewshot_sizes))

    prompt_text_map = build_prompt_text_map()
    score_ranges = build_score_ranges_from_hardcoded()
    global_trait_fallback = build_global_trait_fallback(full_df, TRAIT_COLUMNS)

    summary_rows = []

    for heldout_prompt in heldout_prompts:
        split_prompt_dir = os.path.join(args.split_root, f"heldout_{heldout_prompt}")
        base_ckpt_dir = os.path.join(args.base_root, f"base_prompt{heldout_prompt}", "best_checkpoint")

        if not os.path.isdir(split_prompt_dir):
            print(f"Skipping heldout={heldout_prompt}: split dir not found -> {split_prompt_dir}")
            continue
        if not os.path.isdir(base_ckpt_dir):
            print(f"Skipping heldout={heldout_prompt}: base checkpoint dir not found -> {base_ckpt_dir}")
            continue

        repeat_dirs = sorted(
            [
                os.path.join(split_prompt_dir, d)
                for d in os.listdir(split_prompt_dir)
                if d.startswith("repeat_") and os.path.isdir(os.path.join(split_prompt_dir, d))
            ]
        )

        tokenizer = AutoTokenizer.from_pretrained(args.encoder_name, use_fast=False)

        for repeat_dir in repeat_dirs:
            repeat_name = os.path.basename(repeat_dir)
            print(f"\n=== heldout={heldout_prompt} | {repeat_name} ===")

            dev_df = pd.read_csv(os.path.join(repeat_dir, "dev.tsv"), sep="\t")
            test_df = pd.read_csv(os.path.join(repeat_dir, "test.tsv"), sep="\t")

            dev_dataset = AESDataset(
                df=dev_df,
                tokenizer=tokenizer,
                trait_cols=TRAIT_COLUMNS,
                prompt_col=args.prompt_col,
                text_col=args.text_col,
                prompt_text_map=prompt_text_map,
                score_ranges=score_ranges,
                global_trait_fallback=global_trait_fallback,
                max_length=args.max_length,
            )
            test_dataset = AESDataset(
                df=test_df,
                tokenizer=tokenizer,
                trait_cols=TRAIT_COLUMNS,
                prompt_col=args.prompt_col,
                text_col=args.text_col,
                prompt_text_map=prompt_text_map,
                score_ranges=score_ranges,
                global_trait_fallback=global_trait_fallback,
                max_length=args.max_length,
            )

            dev_loader = DataLoader(
                dev_dataset,
                batch_size=args.eval_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.eval_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )

            zero_model, zero_trait_cols, _ = load_base_model(
                base_ckpt_dir=base_ckpt_dir,
                device=device,
                args=args,
            )
            zero_dev = evaluate(
                model=zero_model,
                dataloader=dev_loader,
                dataset=dev_dataset,
                trait_cols=zero_trait_cols,
                score_ranges=score_ranges,
                global_trait_fallback=global_trait_fallback,
                device=device,
                round_step=args.round_step,
            )
            zero_test = evaluate(
                model=zero_model,
                dataloader=test_loader,
                dataset=test_dataset,
                trait_cols=zero_trait_cols,
                score_ranges=score_ranges,
                global_trait_fallback=global_trait_fallback,
                device=device,
                round_step=args.round_step,
            )

            repeat_out_dir = os.path.join(args.output_root, f"heldout_{heldout_prompt}", repeat_name)
            ensure_dir(repeat_out_dir)
            save_json(zero_dev, os.path.join(repeat_out_dir, "zero_shot_dev_metrics.json"))
            save_json(zero_test, os.path.join(repeat_out_dir, "zero_shot_test_metrics.json"))

            print(format_metrics_for_print("Zero-shot dev", zero_dev))
            print(format_metrics_for_print("Zero-shot test", zero_test))

            for k in fewshot_sizes:
                train_path = os.path.join(repeat_dir, f"fewshot_{k}.tsv")
                if not os.path.exists(train_path):
                    print(f"Skipping k={k}: split not found -> {train_path}")
                    continue

                train_df = pd.read_csv(train_path, sep="\t")
                if len(train_df) == 0:
                    print(f"Skipping k={k}: empty train split")
                    continue

                train_dataset = AESDataset(
                    df=train_df,
                    tokenizer=tokenizer,
                    trait_cols=TRAIT_COLUMNS,
                    prompt_col=args.prompt_col,
                    text_col=args.text_col,
                    prompt_text_map=prompt_text_map,
                    score_ranges=score_ranges,
                    global_trait_fallback=global_trait_fallback,
                    max_length=args.max_length,
                )

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                )

                model, trait_cols, _ = load_base_model(
                    base_ckpt_dir=base_ckpt_dir,
                    device=device,
                    args=args,
                )

                run_dir = os.path.join(repeat_out_dir, f"k_{k}")
                ensure_dir(run_dir)

                model, history, best_epoch, best_dev_qwk = train_one_run(
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

                final_dev = evaluate(
                    model=model,
                    dataloader=dev_loader,
                    dataset=dev_dataset,
                    trait_cols=trait_cols,
                    score_ranges=score_ranges,
                    global_trait_fallback=global_trait_fallback,
                    device=device,
                    round_step=args.round_step,
                )
                final_test = evaluate(
                    model=model,
                    dataloader=test_loader,
                    dataset=test_dataset,
                    trait_cols=trait_cols,
                    score_ranges=score_ranges,
                    global_trait_fallback=global_trait_fallback,
                    device=device,
                    round_step=args.round_step,
                )

                save_json({"history": history}, os.path.join(run_dir, "training_history.json"))
                save_json(final_dev, os.path.join(run_dir, "final_dev_metrics.json"))
                save_json(final_test, os.path.join(run_dir, "final_test_metrics.json"))
                save_json(
                    {
                        "heldout_prompt": heldout_prompt,
                        "repeat_name": repeat_name,
                        "fewshot_k": k,
                        "best_epoch": best_epoch,
                        "best_dev_mean_qwk": best_dev_qwk,
                        "base_checkpoint_dir": base_ckpt_dir,
                        "encoder_name": model.encoder_name,
                        "head_type": model.head_type,
                        "num_bins": model.num_bins,
                        "train_n": len(train_df),
                        "dev_n": len(dev_df),
                        "test_n": len(test_df),
                        "zero_shot_test_mean_qwk": zero_test["mean_qwk"],
                        "zero_shot_test_mean_rmse": zero_test["mean_rmse"],
                    },
                    os.path.join(run_dir, "run_config.json"),
                )

                print(f"\nHead-only | heldout={heldout_prompt} | {repeat_name} | k={k}")
                print(format_metrics_for_print("Final dev", final_dev))
                print(format_metrics_for_print("Final test", final_test))

                summary_rows.append(
                    {
                        "heldout_prompt": heldout_prompt,
                        "repeat_name": repeat_name,
                        "fewshot_k": k,
                        "encoder_name": model.encoder_name,
                        "head_type": model.head_type,
                        "num_bins": model.num_bins,
                        "train_n": len(train_df),
                        "dev_n": len(dev_df),
                        "test_n": len(test_df),
                        "best_epoch": best_epoch,
                        "best_dev_mean_qwk": best_dev_qwk,
                        "final_dev_mean_qwk": final_dev["mean_qwk"],
                        "final_dev_mean_rmse": final_dev["mean_rmse"],
                        "final_test_mean_qwk": final_test["mean_qwk"],
                        "final_test_mean_rmse": final_test["mean_rmse"],
                        "zero_shot_test_mean_qwk": zero_test["mean_qwk"],
                        "zero_shot_test_mean_rmse": zero_test["mean_rmse"],
                    }
                )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(args.output_root, "head_only_summary.csv"), index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
