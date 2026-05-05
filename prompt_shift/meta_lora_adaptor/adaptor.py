import os
import math
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import sys
from pathlib import Path

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
    AESIndexedDataset,
    masked_regression_loss,
    evaluate,
    format_metrics_for_print,
    load_base_checkpoint_into_model,
    apply_lora_to_encoder,
    mark_only_lora_and_head_trainable,
    count_parameters,
    amp_context,
    snapshot_trainable_state,
    restore_trainable_state,
    reptile_meta_update_,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Meta-LoRA adaptation with Reptile-style first-order meta-learning")

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_root", type=str, required=True)
    parser.add_argument("--base_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)

    parser.add_argument("--sep", type=str, default="\t")
    parser.add_argument("--prompt_col", type=str, default="essay_set")
    parser.add_argument("--text_col", type=str, default="essay")

    parser.add_argument("--heldout_prompts", type=str, default="all")
    parser.add_argument("--fewshot_sizes", type=str, default="8,16,32,64,128")

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--round_step", type=float, default=1.0)

    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "huber"])
    parser.add_argument("--huber_delta", type=float, default=1.0)

    # LoRA config for meta-learning.
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, default="query,value")
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
    parser.add_argument("--use_rslora", action="store_true")
    parser.add_argument("--use_dora", action="store_true")
    parser.add_argument("--dropout_override", type=float, default=-1.0)

    # Meta-training config.
    parser.add_argument("--meta_dev_ratio", type=float, default=0.15)
    parser.add_argument("--meta_num_epochs", type=int, default=20)
    parser.add_argument("--meta_episodes_per_epoch", type=int, default=40)
    parser.add_argument("--meta_val_episodes", type=int, default=20)
    parser.add_argument("--meta_support_k", type=int, default=16)
    parser.add_argument("--meta_query_k", type=int, default=32)
    parser.add_argument("--meta_inner_steps", type=int, default=5)
    parser.add_argument("--meta_inner_lr", type=float, default=5e-4)
    parser.add_argument("--meta_step_size", type=float, default=0.1)
    parser.add_argument("--meta_inner_batch_size", type=int, default=8)
    parser.add_argument("--meta_patience", type=int, default=5)

    # Final target adaptation config.
    parser.add_argument("--adapt_num_epochs", type=int, default=30)
    parser.add_argument("--adapt_lr", type=float, default=5e-4)
    parser.add_argument("--adapt_weight_decay", type=float, default=0.01)
    parser.add_argument("--adapt_warmup_ratio", type=float, default=0.1)
    parser.add_argument("--adapt_grad_accum_steps", type=int, default=1)
    parser.add_argument("--adapt_max_grad_norm", type=float, default=1.0)
    parser.add_argument("--adapt_patience", type=int, default=5)

    return parser.parse_args()


def split_prompt_pool_indices(n: int, val_ratio: float, min_train_needed: int, min_val_needed: int, rng: np.random.RandomState):
    perm = rng.permutation(n)
    n_val = max(min_val_needed, int(round(n * val_ratio)))
    if n - n_val < min_train_needed:
        n_val = n - min_train_needed
    if n_val < min_val_needed or (n - n_val) < min_train_needed:
        return None, None
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    return train_idx, val_idx


def build_source_prompt_pools(
    source_df: pd.DataFrame,
    tokenizer,
    prompt_text_map,
    score_ranges,
    global_trait_fallback,
    args,
) -> Tuple[Dict[str, AESDataset], Dict[str, List[int]], Dict[str, List[int]]]:
    prompt_datasets: Dict[str, AESDataset] = {}
    train_pools: Dict[str, List[int]] = {}
    val_pools: Dict[str, List[int]] = {}

    source_prompts = sorted(source_df[args.prompt_col].astype(str).unique().tolist())
    base_seed = args.seed + 12345

    for prompt_id in source_prompts:
        prompt_df = source_df[source_df[args.prompt_col] == prompt_id].copy().reset_index(drop=True)
        if len(prompt_df) == 0:
            continue

        ds = AESDataset(
            df=prompt_df,
            tokenizer=tokenizer,
            trait_cols=TRAIT_COLUMNS,
            prompt_col=args.prompt_col,
            text_col=args.text_col,
            prompt_text_map=prompt_text_map,
            score_ranges=score_ranges,
            global_trait_fallback=global_trait_fallback,
            max_length=args.max_length,
        )

        min_needed = args.meta_support_k + args.meta_query_k
        rng = np.random.RandomState(base_seed + sum(ord(c) for c in prompt_id))
        train_idx, val_idx = split_prompt_pool_indices(
            n=len(ds),
            val_ratio=args.meta_dev_ratio,
            min_train_needed=min_needed,
            min_val_needed=min_needed,
            rng=rng,
        )

        if train_idx is None:
            print(f"Skipping source prompt {prompt_id}: not enough examples for separate meta-train/meta-val pools")
            continue

        prompt_datasets[prompt_id] = ds
        train_pools[prompt_id] = train_idx
        val_pools[prompt_id] = val_idx

    return prompt_datasets, train_pools, val_pools


def sample_support_query(indices: List[int], support_k: int, query_k: int, rng: np.random.RandomState) -> Tuple[List[int], List[int]]:
    if len(indices) < support_k + query_k:
        raise ValueError(f"Not enough examples for an episode: have {len(indices)}, need {support_k + query_k}")
    chosen = rng.choice(indices, size=support_k + query_k, replace=False).tolist()
    return chosen[:support_k], chosen[support_k:]


def inner_adapt_on_dataset(model, dataset, args, device):
    mark_only_lora_and_head_trainable(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.meta_inner_lr, weight_decay=0.0)

    batch_size = min(max(1, args.meta_inner_batch_size), len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    amp_enabled, autocast_device, autocast_dtype = amp_context(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    model.train()
    for _ in range(args.meta_inner_steps):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            label_mask = batch["label_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=amp_enabled):
                preds = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = masked_regression_loss(
                    preds=preds,
                    targets=labels,
                    mask=label_mask,
                    loss_type=args.loss_type,
                    huber_delta=args.huber_delta,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


def evaluate_episode(model, prompt_dataset, query_idx, score_ranges, global_trait_fallback, device, args):
    query_ds = AESIndexedDataset(prompt_dataset, query_idx)
    query_loader = DataLoader(query_ds, batch_size=min(args.eval_batch_size, len(query_ds)), shuffle=False, num_workers=0)
    return evaluate(
        model=model,
        dataloader=query_loader,
        dataset=query_ds,
        trait_cols=TRAIT_COLUMNS,
        score_ranges=score_ranges,
        global_trait_fallback=global_trait_fallback,
        device=device,
        round_step=args.round_step,
        loss_type=args.loss_type,
        huber_delta=args.huber_delta,
    )


def meta_validate(
    model,
    prompt_datasets,
    val_pools,
    score_ranges,
    global_trait_fallback,
    device,
    args,
    rng,
):
    prompts = sorted(val_pools.keys())
    if not prompts:
        return {"mean_qwk": float("nan"), "mean_rmse": float("nan"), "mean_loss": float("nan")}

    qwk_list = []
    rmse_list = []
    loss_list = []

    for _ in range(args.meta_val_episodes):
        prompt_id = rng.choice(prompts)
        support_idx, query_idx = sample_support_query(val_pools[prompt_id], args.meta_support_k, args.meta_query_k, rng)
        start_state = snapshot_trainable_state(model)
        support_ds = AESIndexedDataset(prompt_datasets[prompt_id], support_idx)
        inner_adapt_on_dataset(model, support_ds, args, device)
        metrics = evaluate_episode(model, prompt_datasets[prompt_id], query_idx, score_ranges, global_trait_fallback, device, args)
        restore_trainable_state(model, start_state)

        qwk_list.append(metrics["mean_qwk"])
        rmse_list.append(metrics["mean_rmse"])
        loss_list.append(metrics["loss"])

    valid_qwk = [x for x in qwk_list if not math.isnan(x)]
    valid_rmse = [x for x in rmse_list if not math.isnan(x)]
    valid_loss = [x for x in loss_list if not math.isnan(x)]

    return {
        "mean_qwk": float(np.mean(valid_qwk)) if valid_qwk else float("nan"),
        "mean_rmse": float(np.mean(valid_rmse)) if valid_rmse else float("nan"),
        "mean_loss": float(np.mean(valid_loss)) if valid_loss else float("nan"),
    }


def meta_train_for_prompt(
    heldout_prompt,
    base_ckpt_dir,
    prompt_datasets,
    train_pools,
    val_pools,
    score_ranges,
    global_trait_fallback,
    device,
    args,
    out_dir,
):
    rng = np.random.RandomState(args.seed + sum(ord(c) for c in heldout_prompt))
    lora_target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]

    model, trait_cols, _ = load_base_checkpoint_into_model(
        base_ckpt_dir=base_ckpt_dir,
        device=device,
        dropout_override=args.dropout_override,
    )
    model = apply_lora_to_encoder(
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
    mark_only_lora_and_head_trainable(model)
    param_counts = count_parameters(model)

    prompts = sorted(train_pools.keys())
    if not prompts:
        raise ValueError("No usable source prompt tasks available for meta-training")

    best_path = os.path.join(out_dir, "best_meta_init.pt")
    history = []
    best_score = -1e9
    best_epoch = -1
    early_stop_counter = 0

    for epoch in range(1, args.meta_num_epochs + 1):
        episode_qwks = []
        episode_losses = []
        prog = tqdm(range(args.meta_episodes_per_epoch), desc=f"Meta epoch {epoch}/{args.meta_num_epochs}", leave=False)

        for _ in prog:
            prompt_id = rng.choice(prompts)
            support_idx, query_idx = sample_support_query(train_pools[prompt_id], args.meta_support_k, args.meta_query_k, rng)
            start_state = snapshot_trainable_state(model)
            support_ds = AESIndexedDataset(prompt_datasets[prompt_id], support_idx)
            inner_adapt_on_dataset(model, support_ds, args, device)
            query_metrics = evaluate_episode(model, prompt_datasets[prompt_id], query_idx, score_ranges, global_trait_fallback, device, args)
            episode_qwks.append(query_metrics["mean_qwk"])
            episode_losses.append(query_metrics["loss"])
            reptile_meta_update_(model, start_state, args.meta_step_size)

            valid_qwk = [x for x in episode_qwks if not math.isnan(x)]
            running_qwk = float(np.mean(valid_qwk)) if valid_qwk else float("nan")
            running_loss = float(np.mean([x for x in episode_losses if not math.isnan(x)]))
            prog.set_postfix(qwk=f"{running_qwk:.4f}" if not math.isnan(running_qwk) else "nan", loss=f"{running_loss:.4f}")

        val_metrics = meta_validate(
            model=model,
            prompt_datasets=prompt_datasets,
            val_pools=val_pools,
            score_ranges=score_ranges,
            global_trait_fallback=global_trait_fallback,
            device=device,
            args=args,
            rng=rng,
        )

        history.append({
            "epoch": epoch,
            "train_episode_mean_qwk": float(np.nanmean(episode_qwks)) if episode_qwks else float("nan"),
            "train_episode_mean_loss": float(np.nanmean(episode_losses)) if episode_losses else float("nan"),
            "val_mean_qwk": val_metrics["mean_qwk"],
            "val_mean_rmse": val_metrics["mean_rmse"],
            "val_mean_loss": val_metrics["mean_loss"],
        })

        current_score = val_metrics["mean_qwk"]
        if math.isnan(current_score):
            current_score = -val_metrics["mean_loss"] if not math.isnan(val_metrics["mean_loss"]) else -1e9

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_val_score": best_score,
                    "trait_cols": trait_cols,
                    "param_counts": param_counts,
                    "meta_config": {
                        "support_k": args.meta_support_k,
                        "query_k": args.meta_query_k,
                        "inner_steps": args.meta_inner_steps,
                        "inner_lr": args.meta_inner_lr,
                        "step_size": args.meta_step_size,
                    },
                    "lora_config": {
                        "r": args.lora_r,
                        "alpha": args.lora_alpha,
                        "dropout": args.lora_dropout,
                        "target_modules": lora_target_modules,
                        "bias": args.lora_bias,
                        "use_rslora": args.use_rslora,
                        "use_dora": args.use_dora,
                    },
                },
                best_path,
            )
        else:
            early_stop_counter += 1

        if early_stop_counter >= args.meta_patience:
            break

    save_json({"history": history}, os.path.join(out_dir, "meta_training_history.json"))
    if os.path.exists(best_path):
        best_state = torch.load(best_path, map_location=device)
        model.load_state_dict(best_state["model_state_dict"], strict=True)

    return model, param_counts, best_epoch, best_score


def train_target_adaptation(
    model,
    train_loader,
    dev_loader,
    dev_dataset,
    score_ranges,
    global_trait_fallback,
    device,
    args,
    run_dir,
):
    mark_only_lora_and_head_trainable(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.adapt_lr, weight_decay=args.adapt_weight_decay)

    total_update_steps = max(1, math.ceil(len(train_loader) / args.adapt_grad_accum_steps) * args.adapt_num_epochs)
    warmup_steps = int(total_update_steps * args.adapt_warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    amp_enabled, autocast_device, autocast_dtype = amp_context(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_dev_qwk = -1e9
    best_epoch = -1
    early_stop_counter = 0
    best_ckpt_path = os.path.join(run_dir, "best_meta_lora_adapt.pt")
    history = []

    for epoch in range(1, args.adapt_num_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        num_steps = 0
        progress = tqdm(train_loader, desc=f"Adapt epoch {epoch}/{args.adapt_num_epochs}", leave=False)

        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            label_mask = batch["label_mask"].to(device)

            with torch.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=amp_enabled):
                preds = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = masked_regression_loss(
                    preds=preds,
                    targets=labels,
                    mask=label_mask,
                    loss_type=args.loss_type,
                    huber_delta=args.huber_delta,
                )
                loss = loss / args.adapt_grad_accum_steps

            scaler.scale(loss).backward()

            if step % args.adapt_grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.adapt_max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running_loss += loss.item() * args.adapt_grad_accum_steps
            num_steps += 1
            progress.set_postfix(loss=f"{running_loss / max(num_steps, 1):.4f}")

        train_loss = running_loss / max(num_steps, 1)
        dev_metrics = evaluate(
            model=model,
            dataloader=dev_loader,
            dataset=dev_dataset,
            trait_cols=TRAIT_COLUMNS,
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

        dev_qwk = dev_metrics["mean_qwk"]
        improved = not math.isnan(dev_qwk) and dev_qwk > best_dev_qwk
        if improved:
            best_dev_qwk = dev_qwk
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "best_epoch": best_epoch, "best_dev_mean_qwk": best_dev_qwk},
                best_ckpt_path,
            )
        else:
            early_stop_counter += 1

        if early_stop_counter >= args.adapt_patience:
            break

    if os.path.exists(best_ckpt_path):
        best_state = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_state["model_state_dict"], strict=True)

    return model, history, best_epoch, best_dev_qwk


def build_meta_lora_model_from_base(base_ckpt_dir, device, args):
    model, trait_cols, _ = load_base_checkpoint_into_model(
        base_ckpt_dir=base_ckpt_dir,
        device=device,
        dropout_override=args.dropout_override,
    )
    lora_target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    model = apply_lora_to_encoder(
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
    fewshot_sizes = sorted(parse_int_list(args.fewshot_sizes))

    prompt_text_map = build_prompt_text_map()
    score_ranges = build_score_ranges_from_hardcoded()
    global_trait_fallback = build_global_trait_fallback(full_df, TRAIT_COLUMNS)

    summary_rows = []

    for heldout_prompt in heldout_prompts:
        print(f"\n===== Held-out prompt {heldout_prompt} =====")
        split_prompt_dir = os.path.join(args.split_root, f"heldout_{heldout_prompt}")
        base_ckpt_dir = os.path.join(args.base_root, f"base_prompt{heldout_prompt}", "best_checkpoint")
        meta_out_dir = os.path.join(args.output_root, f"heldout_{heldout_prompt}", "meta_init")
        ensure_dir(meta_out_dir)

        if not os.path.isdir(split_prompt_dir):
            print(f"Skipping heldout={heldout_prompt}: split dir not found -> {split_prompt_dir}")
            continue
        if not os.path.isdir(base_ckpt_dir):
            print(f"Skipping heldout={heldout_prompt}: base checkpoint dir not found -> {base_ckpt_dir}")
            continue

        source_df = full_df[full_df[args.prompt_col] != heldout_prompt].copy().reset_index(drop=True)
        tokenizer = AutoTokenizer.from_pretrained(base_ckpt_dir, use_fast=True)
        prompt_datasets, train_pools, val_pools = build_source_prompt_pools(
            source_df=source_df,
            tokenizer=tokenizer,
            prompt_text_map=prompt_text_map,
            score_ranges=score_ranges,
            global_trait_fallback=global_trait_fallback,
            args=args,
        )

        meta_model, param_counts, meta_best_epoch, meta_best_score = meta_train_for_prompt(
            heldout_prompt=heldout_prompt,
            base_ckpt_dir=base_ckpt_dir,
            prompt_datasets=prompt_datasets,
            train_pools=train_pools,
            val_pools=val_pools,
            score_ranges=score_ranges,
            global_trait_fallback=global_trait_fallback,
            device=device,
            args=args,
            out_dir=meta_out_dir,
        )

        save_json(
            {
                "heldout_prompt": heldout_prompt,
                "meta_best_epoch": meta_best_epoch,
                "meta_best_score": meta_best_score,
                "param_counts": param_counts,
            },
            os.path.join(meta_out_dir, "meta_summary.json"),
        )

        repeat_dirs = sorted(
            [
                os.path.join(split_prompt_dir, d)
                for d in os.listdir(split_prompt_dir)
                if d.startswith("repeat_") and os.path.isdir(os.path.join(split_prompt_dir, d))
            ]
        )

        best_meta_ckpt = torch.load(os.path.join(meta_out_dir, "best_meta_init.pt"), map_location=device)

        for repeat_dir in repeat_dirs:
            repeat_name = os.path.basename(repeat_dir)
            print(f"\n--- heldout={heldout_prompt} | {repeat_name} ---")
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
            dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

            # Base zero-shot for direct comparison with prior methods.
            base_model, base_trait_cols, _ = load_base_checkpoint_into_model(base_ckpt_dir, device, args.dropout_override)
            base_zero_test = evaluate(
                model=base_model,
                dataloader=test_loader,
                dataset=test_dataset,
                trait_cols=base_trait_cols,
                score_ranges=score_ranges,
                global_trait_fallback=global_trait_fallback,
                device=device,
                round_step=args.round_step,
                loss_type=args.loss_type,
                huber_delta=args.huber_delta,
            )

            # Meta-initialization zero-shot, before target adaptation.
            meta_init_model, meta_trait_cols = build_meta_lora_model_from_base(base_ckpt_dir, device, args)
            meta_init_model.load_state_dict(best_meta_ckpt["model_state_dict"], strict=True)
            meta_init_dev = evaluate(
                model=meta_init_model,
                dataloader=dev_loader,
                dataset=dev_dataset,
                trait_cols=meta_trait_cols,
                score_ranges=score_ranges,
                global_trait_fallback=global_trait_fallback,
                device=device,
                round_step=args.round_step,
                loss_type=args.loss_type,
                huber_delta=args.huber_delta,
            )
            meta_init_test = evaluate(
                model=meta_init_model,
                dataloader=test_loader,
                dataset=test_dataset,
                trait_cols=meta_trait_cols,
                score_ranges=score_ranges,
                global_trait_fallback=global_trait_fallback,
                device=device,
                round_step=args.round_step,
                loss_type=args.loss_type,
                huber_delta=args.huber_delta,
            )

            repeat_out_dir = os.path.join(args.output_root, f"heldout_{heldout_prompt}", repeat_name)
            ensure_dir(repeat_out_dir)
            save_json(base_zero_test, os.path.join(repeat_out_dir, "base_zero_shot_test_metrics.json"))
            save_json(meta_init_dev, os.path.join(repeat_out_dir, "meta_init_dev_metrics.json"))
            save_json(meta_init_test, os.path.join(repeat_out_dir, "meta_init_test_metrics.json"))

            print(format_metrics_for_print("Base zero-shot test", base_zero_test))
            print(format_metrics_for_print("Meta-init zero-shot test", meta_init_test))

            for k in fewshot_sizes:
                train_path = os.path.join(repeat_dir, f"fewshot_{k}.tsv")
                if not os.path.exists(train_path):
                    print(f"Skipping k={k}: split not found -> {train_path}")
                    continue
                train_df = pd.read_csv(train_path, sep="\t")
                if len(train_df) == 0:
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
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

                model, trait_cols = build_meta_lora_model_from_base(base_ckpt_dir, device, args)
                model.load_state_dict(best_meta_ckpt["model_state_dict"], strict=True)
                run_dir = os.path.join(repeat_out_dir, f"k_{k}")
                ensure_dir(run_dir)

                model, adapt_history, best_epoch, best_dev_qwk = train_target_adaptation(
                    model=model,
                    train_loader=train_loader,
                    dev_loader=dev_loader,
                    dev_dataset=dev_dataset,
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
                    loss_type=args.loss_type,
                    huber_delta=args.huber_delta,
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
                    loss_type=args.loss_type,
                    huber_delta=args.huber_delta,
                )

                save_json({"history": adapt_history}, os.path.join(run_dir, "adapt_training_history.json"))
                save_json(final_dev, os.path.join(run_dir, "final_dev_metrics.json"))
                save_json(final_test, os.path.join(run_dir, "final_test_metrics.json"))
                save_json(
                    {
                        "heldout_prompt": heldout_prompt,
                        "repeat_name": repeat_name,
                        "fewshot_k": k,
                        "meta_best_epoch": meta_best_epoch,
                        "meta_best_score": meta_best_score,
                        "best_adapt_epoch": best_epoch,
                        "best_dev_mean_qwk": best_dev_qwk,
                        "base_zero_shot_test_mean_qwk": base_zero_test["mean_qwk"],
                        "base_zero_shot_test_mean_rmse": base_zero_test["mean_rmse"],
                        "meta_init_test_mean_qwk": meta_init_test["mean_qwk"],
                        "meta_init_test_mean_rmse": meta_init_test["mean_rmse"],
                        "trainable_params": param_counts["trainable"],
                        "total_params": param_counts["total"],
                    },
                    os.path.join(run_dir, "run_config.json"),
                )

                print(f"\nMeta-LoRA | heldout={heldout_prompt} | {repeat_name} | k={k}")
                print(format_metrics_for_print("Final dev", final_dev))
                print(format_metrics_for_print("Final test", final_test))

                summary_rows.append(
                    {
                        "heldout_prompt": heldout_prompt,
                        "repeat_name": repeat_name,
                        "fewshot_k": k,
                        "train_n": len(train_df),
                        "dev_n": len(dev_df),
                        "test_n": len(test_df),
                        "meta_best_epoch": meta_best_epoch,
                        "best_epoch": best_epoch,
                        "best_dev_mean_qwk": best_dev_qwk,
                        "final_dev_mean_qwk": final_dev["mean_qwk"],
                        "final_dev_mean_rmse": final_dev["mean_rmse"],
                        "final_test_mean_qwk": final_test["mean_qwk"],
                        "final_test_mean_rmse": final_test["mean_rmse"],
                        "base_zero_shot_test_mean_qwk": base_zero_test["mean_qwk"],
                        "base_zero_shot_test_mean_rmse": base_zero_test["mean_rmse"],
                        "meta_init_test_mean_qwk": meta_init_test["mean_qwk"],
                        "meta_init_test_mean_rmse": meta_init_test["mean_rmse"],
                        "trainable_params": param_counts["trainable"],
                        "total_params": param_counts["total"],
                    }
                )

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(os.path.join(args.output_root, "meta_lora_summary.csv"), index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
