#!/bin/bash
#SBATCH --job-name=AES_LoRA_META
#SBATCH -o aes_lora_meta.out
#SBATCH -e aes_lora_meta.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sarathka@staff.vuw.ac.nz

set -euo pipefail
set -x

trap 'echo "Error at line $LINENO"; exit 1' ERR

module load GCCcore/11.2.0
module load Python/3.9.6

source mytest/bin/activate
export PYTHONUNBUFFERED=1

python3 -u lora_adapt_aes.py \
  --folds_dir ../data/vuw_data_folds_v2\
  --fold_id 1 \
  --train_file_template fold_{fold}/train.tsv \
  --dev_file_template fold_{fold}/dev.tsv \
  --test_file_template fold_{fold}/test.tsv \
  --sep $'\t' \
  --base_model_path results_finetune_baseline/best_alignment_model \
  --meta_checkpoint results_meta_fomaml_run4/best_meta_model/meta_model.pt \
  --output_dir results_lora_meta_adapt/fold1 \
  --batch_size 8 \
  --num_epochs 20 \
  --lr 2e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --min_score 1 \
  --max_score 6