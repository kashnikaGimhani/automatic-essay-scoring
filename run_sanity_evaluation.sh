#!/bin/bash
set -e

# (Optional) activate your venv if you have one
# source aes_env/bin/activate

# Paths (edit these if needed)
EVAL_SCRIPT="model_evaluator.py"
DATA_PATH="data/5fold_cv/fold_1"
MODEL_PATH="./results/fold_1/checkpoint-250"      # <-- change to your checkpoint folder
OUT_PATH="./outputs/sanity_eval_fold_1"           # creates output file: sanity_eval_fold_1.csv
SANITY_SPLIT="train[:5%]"

# Sanity settings (fast)
MAX_TGT_LEN=256
TEST_BATCH=2

python3 "$EVAL_SCRIPT" \
  --data_path "$DATA_PATH" \
  --model_path "$MODEL_PATH" \
  --output_path "$OUT_PATH" \
  --max_tgt_len $MAX_TGT_LEN \
  --test_batch $TEST_BATCH 
