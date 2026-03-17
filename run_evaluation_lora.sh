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

FOLD_DIR="./data/vuw_data_folds/fold_1"
LORA_CKPT="./results_small_lora_asaptraits/fold_1/checkpoint-300"  
BASE_MODEL="./results/fold_1/checkpoint-10000"                      
OUT_CSV="./outputs/lora_vuw_data_fold_1_outputs.csv"

python3 model_evaluator_lora.py \
  --data_path "${FOLD_DIR}" \
  --model_path "${LORA_CKPT}" \
  --base_model "${BASE_MODEL}" \
  --test_batch 2 \
  --max_new_tokens 64 \
  --output_path "${OUT_CSV}"