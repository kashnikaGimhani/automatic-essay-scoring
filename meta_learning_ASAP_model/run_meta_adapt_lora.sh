#!/bin/bash
#SBATCH --job-name=AES_EPI_LORA
#SBATCH -o aes_epi_lora.out
#SBATCH -e aes_epi_lora.err
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

python3 -u meta_adapt_lora.py \
  --folds_dir ../data/vuw_data_folds_v2 \
  --fold_id 2 \
  --train_file_template fold_{fold}/train.tsv \
  --dev_file_template fold_{fold}/dev.tsv \
  --test_file_template fold_{fold}/test.tsv \
  --sep $'\t' \
  --base_model_path results_finetune_baseline/best_alignment_model \
  --meta_checkpoint results_meta_fomaml_run4/best_meta_model/meta_model.pt \
  --output_dir results_epi_lora_adapt/fold2 \
  --num_epochs 20 \
  --lr 1e-4 \
  --inner_lr 1e-3 \
  --support_size 8 \
  --query_size 16 \
  --tasks_per_episode 4 \
  --episodes_per_epoch 100 \
  --inner_steps 1 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --eval_batch_size 16 \
  --min_score 1 \
  --max_score 6 \
  --sample_with_replacement