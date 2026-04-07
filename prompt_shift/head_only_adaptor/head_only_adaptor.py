import os
import math
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from utils import (
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
    AESDataset,
    MultiTraitAESModel,
    freeze_encoder_only,
    masked_regression_loss,
    evaluate,
    format_metrics_for_print,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Head-only adaptation on reusable few-shot target splits")

    parser.add_argument("--data_path", type=str, required=True, help="Full original dataset; used only for global fallback ranges")
    parser.add_argument("--split_root", type=str, required=True, help="Root directory created by create_target_fewshot_splits.py")
    parser.add_argument("--base_root", type=str, required=True, help='Root containing base models, e.g. base_root/heldout_1/best_checkpoint')
    parser.add_argument("--output_root", type=str, required=True)

    parser.add_argument("--sep", type=str, default="\t")
    parser.add_argument("--prompt_col", type=str, default="essay_set")
    parser.add_argument("--text_col", type=str, default="essay")

    parser.add_argument("--heldout_prompts", type=str, default="all")
    parser.add_argument("--fewshot_sizes", type=str, default="8,16,32,64,128")

    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3, help="Head-only LR can usually be higher than full FT")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dropout_override", type=float, default=-1.0, help="Use -1 to keep base checkpoint dropout")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "huber"])
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--round_step", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_base_model(base_ckpt_dir: str, device: torch.device, dropout_override: float):
    ckpt_path = os.path.join(base_ckpt_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Base checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    trait_cols = ckpt.get("trait_cols", TRAIT_COLUMNS)
    dropout = ckpt.get("dropout", 0.1)
    if dropout_override >= 0:
        dropout = dropout_override

    model = MultiTraitAESModel(
        dropout=dropout,
        num_traits=len(trait_cols),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    return model, trait_cols, ckpt


def train_one_run(
    model,
    train_loader,
    dev_loader,
    train_dataset,
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

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

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

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
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
            loss_type=args.loss_type,
            huber_delta=args.huber_delta,
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
        base_ckpt_dir = os.path.join(args.base_root, f"heldout_{heldout_prompt}", "best_checkpoint")

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

        for repeat_dir in repeat_dirs:
            repeat_name = os.path.basename(repeat_dir)
            print(f"\n=== heldout={heldout_prompt} | {repeat_name} ===")

            dev_df = pd.read_csv(os.path.join(repeat_dir, "dev.tsv"), sep="\t")
            test_df = pd.read_csv(os.path.join(repeat_dir, "test.tsv"), sep="\t")

            tokenizer = AutoTokenizer.from_pretrained(base_ckpt_dir, use_fast=True)

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

            # Optional zero-shot baseline before adaptation, saved once per repeat.
            zero_model, zero_trait_cols, _ = load_base_model(
                base_ckpt_dir=base_ckpt_dir,
                device=device,
                dropout_override=args.dropout_override,
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
                loss_type=args.loss_type,
                huber_delta=args.huber_delta,
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
                loss_type=args.loss_type,
                huber_delta=args.huber_delta,
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

                model, trait_cols, base_ckpt_meta = load_base_model(
                    base_ckpt_dir=base_ckpt_dir,
                    device=device,
                    dropout_override=args.dropout_override,
                )

                run_dir = os.path.join(repeat_out_dir, f"k_{k}")
                ensure_dir(run_dir)

                model, history, best_epoch, best_dev_qwk = train_one_run(
                    model=model,
                    train_loader=train_loader,
                    dev_loader=dev_loader,
                    train_dataset=train_dataset,
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

                save_json(history and {"history": history} or {"history": []}, os.path.join(run_dir, "training_history.json"))
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
    from torch.utils.data import DataLoader
    main()