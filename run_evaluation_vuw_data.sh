#!/bin/bash

#
#SBATCH --job-name=AES_Evaluation_vuw_data
#SBATCH -o evaluation_vuw_data.out
#SBATCH -e evaluation_vuw_data.err
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
EVAL_SCRIPT="model_evaluator_vuw_data.py"

# -------- Paths --------
DATA_PATH="data/vuw_data/113.tsv"
MODEL_PATH="./results/fold_1/checkpoint-10000"
OUT_PATH="./outputs/eval_vuw_data"

# -------- Parameters --------
MAX_TGT_LEN=168
TRAIN_BATCH=4
VALID_BATCH=4
TEST_BATCH=4
TRAITS="traits"

-------- Run evaluation --------
python3 "$EVAL_SCRIPT" \
  --data_file "$DATA_PATH" \
  --model_path "$MODEL_PATH" \
  --output_path "$OUT_PATH" \
  --max_new_tokens $MAX_TGT_LEN \
  --test_batch $TEST_BATCH \
  --weight "quadratic" \
  --max_input_len 768 \
  #--train_batch $TRAIN_BATCH \
  #--valid_batch $VALID_BATCH \