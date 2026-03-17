#!/usr/bin/env python3
"""
Final LoRA evaluation script (LoRA model only) for your small dataset with ASAP trait names.

Fix:
- Correctly loads LoRA adapter checkpoints (adapter_config.json present) by requiring --base_model.
- Still supports loading a full merged model (pytorch_model.bin present) without --base_model.

Run (adapter checkpoint):
python model_evaluator_lora_small_asaptraits_final.py \
  --data_path ./small_folds/fold_1 \
  --model_path ./results_small_lora_asaptraits/fold_1/checkpoint-XXXX \
  --base_model ./results/fold_1/checkpoint-10000 \
  --output_path ./eval_small/fold_1_outputs.csv

Run (merged/full checkpoint):
python model_evaluator_lora_small_asaptraits_final.py \
  --data_path ./small_folds/fold_1 \
  --model_path /path/to/merged_model_dir
"""

import argparse
import os
import re
import json
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq
from typing import Optional

# ---- Use your already-written evaluation utils ----
from utils.eval_qwk_lora import (
    Evaluator,
    ASAP_TRAITS_WITH_OVERALL,
    build_asap_target_strings_from_rows,
    ASAP_TRAITS_5,
)


def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# -----------------------------
# (0) Robust LoRA loading
# -----------------------------
def is_lora_adapter_dir(path: str) -> bool:
    return os.path.exists(os.path.join(path, "adapter_config.json"))


def load_model_and_tokenizer(model_path: str, base_model: Optional[str]):
    """
    If model_path is a LoRA adapter dir:
        tokenizer/base are loaded from base_model
        adapter is loaded from model_path
    Else:
        load full model/tokenizer from model_path
    """
    if is_lora_adapter_dir(model_path):
        if not base_model:
            raise ValueError(
                "Detected LoRA adapter checkpoint (adapter_config.json found). "
                "You must pass --base_model pointing to your ASAP-pretrained T5 checkpoint."
            )
        from peft import PeftModel

        tokenizer = T5Tokenizer.from_pretrained(base_model)
        base = T5ForConditionalGeneration.from_pretrained(base_model)
        model = PeftModel.from_pretrained(base, model_path)

        # Optional but recommended: merge adapter into base for faster inference
        try:
            model = model.merge_and_unload()
        except Exception:
            pass

        return model, tokenizer

    # Full merged model
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    return model, tokenizer


# -----------------------------
# (1) Normalize model text output -> dict-like string
# -----------------------------
def normalize_pred_to_dictlike(text: str) -> str:
    """
    Converts:
      "content=4; sentence_fluency=5; ..." -> "{content: 4, sentence_fluency: 5, ...}"
    so Evaluator.read_results can parse it.
    """
    if text is None:
        return "{}"
    s = str(text).strip()

    if s.startswith("{") and (":" in s):
        return s

    s = re.sub(r"(\b[a-zA-Z_]+\b)\s*=\s*", r"\1: ", s)
    s = s.replace(";", ",")
    s = re.sub(r"\s*,\s*", ", ", s)

    if ":" in s and not s.lstrip().startswith("{"):
        s = "{" + s + "}"
    return s


# -----------------------------
# (2) Compute overall from the 5 predicted traits
# -----------------------------
def parse_5_traits_from_dictlike(s: str) -> dict:
    out = {}
    s_low = (s or "").lower()
    for k in ASAP_TRAITS_5:
        m = re.search(rf"{re.escape(k)}\s*[:=]\s*([0-9]+(\.[0-9]+)?)", s_low)
        if m:
            out[k] = float(m.group(1))
    return out


def add_computed_overall(pred_dictlike: str) -> str:
    scores = parse_5_traits_from_dictlike(pred_dictlike)
    if all(k in scores for k in ASAP_TRAITS_5):
        vals = [scores[k] for k in ASAP_TRAITS_5]
        overall = int(round(sum(vals) / len(vals)))
        scores["overall"] = float(overall)
    return json.dumps(scores, ensure_ascii=False)


# -----------------------------
# (3) Tokenize inputs for generation
# -----------------------------
def preprocess_inputs(tokenizer, examples):
    inputs = []
    for i in range(len(examples["essay"])):
        essay = examples["essay"][i] or ""
        inp = (
            "score traits (content, sentence_fluency, organization, word_choice, conventions) "
            "for this essay: " + str(essay)
        )
        inputs.append(inp)

    return tokenizer(
        inputs,
        max_length=512,
        padding="max_length",
        truncation=True,
    )


def generate_predictions(model, tokenizer, dataset, batch_size, max_new_tokens, device):
    dataset = dataset.with_format("torch", columns=["input_ids", "attention_mask"])
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    preds_txt = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items() if v is not None}
            gen = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
                use_cache=True,
            )
            preds_txt.extend(tokenizer.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True))
    return preds_txt


def main():
    ap = argparse.ArgumentParser(description="Evaluate LoRA fine-tuned T5 on small dataset (ASAP trait names, overall computed).")
    ap.add_argument("--data_path", type=str, required=True, help="Fold dir containing test.tsv")
    ap.add_argument("--model_path", type=str, required=True, help="LoRA adapter checkpoint dir OR merged full model dir")
    ap.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="REQUIRED if model_path is a LoRA adapter dir. Path to ASAP-pretrained T5 checkpoint.",
    )
    ap.add_argument("--test_batch", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--output_path", type=str, default=None, help="CSV path to save raw+normalized predictions")
    args = ap.parse_args()

    test_file = os.path.join(args.data_path, "test.tsv")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Could not find {test_file}")

    # (A) Load test as dataframe and build targets via your helper
    df_test = pd.read_csv(test_file, sep="\t")
    targets_list = build_asap_target_strings_from_rows(df_test).tolist()

    # (B) Load model/tokenizer correctly (adapter vs full)
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model)

    device = detect_device()
    model = model.to(device)

    # (C) Prepare generation inputs
    test_raw = load_dataset("csv", data_files=test_file, delimiter="\t")["train"]
    test_inputs = test_raw.map(lambda ex: preprocess_inputs(tokenizer, ex), batched=True)

    # (D) Generate
    preds_raw = generate_predictions(model, tokenizer, test_inputs, args.test_batch, args.max_new_tokens, device)

    # (E) Normalize preds, compute overall, convert to JSON dict strings
    preds_dictlike = [normalize_pred_to_dictlike(p) for p in preds_raw]
    preds_json = [add_computed_overall(p) for p in preds_dictlike]

    print("\n--- DEBUG SAMPLES ---")
    for i in range(5):
        print(f"\nRow {i}")
        print("RAW :", preds_raw[i])
        print("DICT:", preds_dictlike[i])
        print("JSON:", preds_json[i])
        print("TGT :", targets_list[i])

    # (F) Evaluate using your Evaluator
    evaluator = Evaluator(ASAP_TRAITS_WITH_OVERALL)
    qwk_results = evaluator.evaluate_notnull(preds_json, targets_list)

    # (G) Save outputs
    out_path = args.output_path or os.path.join(args.model_path, "test_outputs.csv")
    pd.DataFrame(
        {
            "predictions_raw": preds_raw,
            "predictions_dictlike": preds_dictlike,
            "predictions_json_with_overall": preds_json,
            "targets_json_with_overall": targets_list,
        }
    ).to_csv(out_path, index=False)

    qwk_path = os.path.join(args.model_path, "qwk_results_small.csv")
    pd.DataFrame([qwk_results]).to_csv(qwk_path, index=False)

    print("✅ Saved predictions:", out_path)
    print("✅ Saved QWK results:", qwk_path)
    print("\nQWK results:")
    for k, v in qwk_results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()