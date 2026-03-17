#!/usr/bin/env python3
"""
LoRA fine-tuning for T5 on your small dataset, using ASAP trait names.

Changes vs your earlier version:
- Trait keys in targets match ASAP: content, sentence_fluency, organization, word_choice, conventions
- No overall supervision: overall is NOT in the target string (computed later from predicted traits)
- No range conversion: your traits are already 1–6 and ASAP base was trained on 1–6
"""

import argparse
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType


# ---- (1) Trait mapping: small dataset -> ASAP trait names ----
# Your requirement:
# ideas -> content
# flow -> sentence_fluency
# coherence -> organization
# vocab -> word_choice
# grammar -> conventions
ASAP_TRAITS = ["content", "sentence_fluency", "organization", "word_choice", "conventions"]


def build_target_asap(ex: dict) -> str:
    """
    Build target string ONLY for 5 ASAP traits.
    IMPORTANT: We DO NOT include overall in training target.
    """
    return (
        f"content={ex['ideas']}; "
        f"sentence_fluency={ex['flow']}; "
        f"organization={ex['coherence']}; "
        f"word_choice={ex['vocab']}; "
        f"conventions={ex['grammar']}"
    )


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune T5 for AES (small dataset, ASAP trait names, no overall)")
    parser.add_argument("--max_tgt_len", type=int, default=64)
    parser.add_argument("--train_batch", type=int, default=2)
    parser.add_argument("--valid_batch", type=int, default=2)
    parser.add_argument("--output_path", type=str, default="./results_small_lora_asaptraits")
    parser.add_argument("--data_path", type=str, required=True, help="Fold dir containing train.tsv/dev.tsv")
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--steps", type=int, default=300, help="eval/save steps")

    # Base model (your ASAP-pretrained checkpoint)
    # Keep as an arg so you can change folds/checkpoints easily
    parser.add_argument("--base_model", type=str, required=True, help="Path to ASAP-pretrained checkpoint directory")

    # LoRA knobs
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=22)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # optimization knobs
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    args = parser.parse_args()

    # ---- (2) Load fold train/dev data (TSV) ----
    dataset = load_dataset("csv", data_files=f"{args.data_path}/train.tsv", delimiter="\t")
    val_set = load_dataset("csv", data_files=f"{args.data_path}/dev.tsv", delimiter="\t")
    dataset["valid"] = val_set["train"]

    # ---- (3) Load ASAP-pretrained tokenizer + model ----
    tokenizer = T5Tokenizer.from_pretrained(args.base_model)
    base_model = T5ForConditionalGeneration.from_pretrained(args.base_model)

    # ---- (4) Attach LoRA adapters (freezes base weights internally) ----
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q", "v"],  # typical for T5 attention projections
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)
    for name, param in model.named_parameters():
        if "lm_head" in name:
            param.requires_grad = True
    model.print_trainable_parameters()

    # ---- (5) Tokenization: instruction input + mapped ASAP target ----
    def preprocess_function(examples):
        inputs, targets = [], []
        for i in range(len(examples["essay"])):
            essay = examples["essay"][i]
            if not essay:
                continue

            # Keep instruction stable across training/eval
            inp = (
                "score traits (content, sentence_fluency, organization, word_choice, conventions) "
                "for this essay: " + essay
            )
            inputs.append(inp)

            ex = {
                "ideas": examples["ideas"][i],
                "flow": examples["flow"][i],
                "coherence": examples["coherence"][i],
                "vocab": examples["vocab"][i],
                "grammar": examples["grammar"][i],
            }
            targets.append(build_target_asap(ex))

        model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)
        labels = tokenizer(text_target=targets, max_length=args.max_tgt_len, padding="max_length", truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_set = dataset["train"].map(preprocess_function, batched=True)
    valid_set = dataset["valid"].map(preprocess_function, batched=True)

    # ---- (6) Trainer args: early stopping + generate during eval ----
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_path,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.valid_batch,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_dir=f"{args.output_path}/logs",
        report_to="tensorboard",
        logging_strategy="steps",
        logging_steps=50,
        eval_steps=args.steps,
        save_steps=args.steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        generation_max_length=args.max_tgt_len,
        predict_with_generate=True,
        save_total_limit=2,
    )

    # ---- (7) Train + eval ----
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    main()