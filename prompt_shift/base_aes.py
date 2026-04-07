import os
import math
import argparse

import pandas as pd
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from utils import (
    MODEL_NAME,
    TRAIT_COLUMNS,
    set_seed,
    ensure_dir,
    save_json,
    normalize_prompt_id,
    build_prompt_text_map,
    build_score_ranges_from_hardcoded,
    build_global_trait_fallback,
    split_source_by_prompt,
    AESDataset,
    MultiTraitAESModel,
    masked_regression_loss,
    evaluate,
    format_metrics_for_print,
)


print("Libraries imported successfully.", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Base training for prompt-shift AES")

    parser.add_argument("--data_path", type=str, required=True, help="Path to TSV/CSV data file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--sep", type=str, default="\t", help="File separator, default is tab")

    parser.add_argument("--heldout_prompt", type=str, required=True, help="Prompt ID to hold out")
    parser.add_argument("--prompt_col", type=str, default="essay_set")
    parser.add_argument("--text_col", type=str, default="essay")
    parser.add_argument("--id_col", type=str, default="essay_id")

    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=3)

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


def save_checkpoint(
    output_dir: str,
    model: torch.nn.Module,
    tokenizer,
    args,
    trait_cols,
    score_ranges,
    prompt_text_map,
    train_metrics,
    dev_metrics,
):
    ensure_dir(output_dir)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_name": MODEL_NAME,
        "trait_cols": trait_cols,
        "dropout": args.dropout,
        "max_length": args.max_length,
        "loss_type": args.loss_type,
        "huber_delta": args.huber_delta,
        "heldout_prompt": args.heldout_prompt,
    }
    torch.save(ckpt, os.path.join(output_dir, "best_model.pt"))

    tokenizer.save_pretrained(output_dir)
    model.encoder.config.save_pretrained(output_dir)

    save_json(score_ranges, os.path.join(output_dir, "score_ranges.json"))
    save_json(prompt_text_map, os.path.join(output_dir, "prompt_texts.json"))
    save_json(
        {
            "model_name": MODEL_NAME,
            "heldout_prompt": args.heldout_prompt,
            "trait_cols": trait_cols,
            "prompt_col": args.prompt_col,
            "text_col": args.text_col,
            "id_col": args.id_col,
            "max_length": args.max_length,
            "loss_type": args.loss_type,
            "huber_delta": args.huber_delta,
            "round_step": args.round_step,
            "dev_ratio": args.dev_ratio,
            "seed": args.seed,
        },
        os.path.join(output_dir, "training_config.json"),
    )

    save_json(train_metrics, os.path.join(output_dir, "last_train_metrics.json"))
    save_json(dev_metrics, os.path.join(output_dir, "best_dev_metrics.json"))


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    df = pd.read_csv(args.data_path, sep=args.sep)
    df[args.prompt_col] = df[args.prompt_col].apply(normalize_prompt_id)

    trait_cols = TRAIT_COLUMNS
    print("Trait columns:", trait_cols)

    for trait in trait_cols:
        if trait in df.columns:
            df[trait] = pd.to_numeric(df[trait], errors="coerce")

    heldout_prompt = normalize_prompt_id(args.heldout_prompt)
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

    print(f"All prompts      : {all_prompts}")
    print(f"Held-out prompt  : {heldout_prompt}")
    print(f"Source train size: {len(train_df)}")
    print(f"Source dev size  : {len(dev_df)}")
    print(f"Held-out size    : {len(target_df)}")

    prompt_text_map = build_prompt_text_map()
    score_ranges = build_score_ranges_from_hardcoded()
    global_trait_fallback = build_global_trait_fallback(df, trait_cols)

    if args.save_split_files:
        split_dir = os.path.join(args.output_dir, "splits")
        ensure_dir(split_dir)
        train_df.to_csv(os.path.join(split_dir, "source_train.tsv"), sep="\t", index=False)
        dev_df.to_csv(os.path.join(split_dir, "source_dev.tsv"), sep="\t", index=False)
        target_df.to_csv(os.path.join(split_dir, f"heldout_prompt_{heldout_prompt}.tsv"), sep="\t", index=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    train_dataset = AESDataset(
        df=train_df,
        tokenizer=tokenizer,
        trait_cols=trait_cols,
        prompt_col=args.prompt_col,
        text_col=args.text_col,
        prompt_text_map=prompt_text_map,
        score_ranges=score_ranges,
        global_trait_fallback=global_trait_fallback,
        max_length=args.max_length,
    )
    dev_dataset = AESDataset(
        df=dev_df,
        tokenizer=tokenizer,
        trait_cols=trait_cols,
        prompt_col=args.prompt_col,
        text_col=args.text_col,
        prompt_text_map=prompt_text_map,
        score_ranges=score_ranges,
        global_trait_fallback=global_trait_fallback,
        max_length=args.max_length,
    )
    target_dataset = AESDataset(
        df=target_df,
        tokenizer=tokenizer,
        trait_cols=trait_cols,
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

    model = MultiTraitAESModel(
        dropout=args.dropout,
        num_traits=len(trait_cols),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_update_steps = math.ceil(len(train_loader) / args.grad_accum_steps) * args.num_epochs
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    best_dev_qwk = -1e9
    best_epoch = -1
    early_stop_counter = 0
    last_train_metrics = {}

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        num_steps = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}", leave=True)

        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            label_mask = batch["label_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device) if "token_type_ids" in batch else None

            with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                preds = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
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

        dev_metrics = evaluate(
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

        print()
        print(f"Epoch {epoch} finished")
        print(f"Train loss: {train_loss:.6f}")
        print(format_metrics_for_print("Source Dev", dev_metrics))
        print()

        dev_qwk = dev_metrics["mean_qwk"]
        improved = not math.isnan(dev_qwk) and dev_qwk > best_dev_qwk

        if improved:
            best_dev_qwk = dev_qwk
            best_epoch = epoch
            early_stop_counter = 0

            best_dir = os.path.join(args.output_dir, "best_checkpoint")
            save_checkpoint(
                output_dir=best_dir,
                model=model,
                tokenizer=tokenizer,
                args=args,
                trait_cols=trait_cols,
                score_ranges=score_ranges,
                prompt_text_map=prompt_text_map,
                train_metrics=last_train_metrics,
                dev_metrics=dev_metrics,
            )
            print(f"Saved new best checkpoint to: {best_dir}")
        else:
            early_stop_counter += 1
            print(f"No improvement. Early stop counter: {early_stop_counter}/{args.patience}")

        if early_stop_counter >= args.patience:
            print("Early stopping triggered.")
            break

    print(f"Best epoch: {best_epoch}")
    print(f"Best source-dev mean QWK: {best_dev_qwk:.6f}")

    best_ckpt_path = os.path.join(args.output_dir, "best_checkpoint", "best_model.pt")
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

    final_dev_metrics = evaluate(
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
    print(format_metrics_for_print("Final Source Dev", final_dev_metrics))

    if args.run_zero_shot_eval and len(target_dataset) > 0:
        zero_shot_metrics = evaluate(
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
        print(format_metrics_for_print(f"Zero-shot Held-out Prompt {heldout_prompt}", zero_shot_metrics))

    print("\nDone.")


if __name__ == "__main__":
    main()
