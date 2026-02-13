#!/bin/bash

#
#SBATCH --job-name=AES_Evaluation
#SBATCH -o evaluation.out
#SBATCH -e evaluation.err
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sarathka@staff.vuw.ac.nz

module load GCCcore/11.2.0
module load Python/3.9.6

source mytest/bin/activate

# -------- Script --------
EVAL_SCRIPT="model_evaluator.py"

# -------- Paths --------
DATA_PATH="data/5fold_cv/fold_1"
MODEL_PATH="./results/fold_1/checkpoint-10000"
OUT_PATH="./outputs/eval_fold_1"

# -------- Parameters --------
MAX_TGT_LEN=168
TRAIN_BATCH=4
VALID_BATCH=4
TEST_BATCH=4
TRAITS="traits"

-------- Run evaluation --------
python3 "$EVAL_SCRIPT" \
  --data_path "$DATA_PATH" \
  --model_path "$MODEL_PATH" \
  --output_path "$OUT_PATH" \
  --max_tgt_len $MAX_TGT_LEN \
  --test_batch $TEST_BATCH
  #--train_batch $TRAIN_BATCH \
  #--valid_batch $VALID_BATCH \