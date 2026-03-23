#!/bin/bash
#
#SBATCH --job-name=AES_Meta_Training
#SBATCH -o meta_training.out
#SBATCH -e meta_training.err
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sarathka@staff.vuw.ac.nz

# Strict error handling
set -euo pipefail
set -x

# Custom error message
trap 'echo "❌ Error at line $LINENO"; exit 1' ERR

echo "Starting job..."
echo "Working dir: $(pwd)"


module load GCCcore/11.2.0
module load Python/3.9.6

source mytest/bin/activate

export PYTHONUNBUFFERED=1

echo "Starting python script running..."

python3 -u meta_train_aes_v2.py \
  --train_file ../data/asap_train_with_all_traits.tsv \
  --model_path results_finetune_baseline/best_alignment_model \
  --output_dir results_meta_aes_v2 \
  --train_prompts 1 2 3 4 5 6 \
  --val_prompts 7 \
  --test_prompts 8 \
  --trainable_mode last_k \
  --inner_lr 1e-3 \
  --meta_lr 5e-5 \
  --support_size 8 \
  --query_size 16 \
  --tasks_per_meta_batch 4 \
  --meta_steps 1000 \
  --val_every 100 \
  --val_episodes 100 \
  --test_episodes 200 \
  --log_qwk_details \
  --qwk_log_top_k 20

echo "Job completed successfully!"