#!/bin/bash
#
#SBATCH --job-name=AES_Meta_Training_FOMAML
#SBATCH -o meta_training_fomaml.out
#SBATCH -e meta_training_fomaml.err
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

python3 -u meta_train_fomaml.py \
  --train_file ../data/asap_train_with_all_traits.tsv \
  --model_path results_finetune_baseline/best_alignment_model \
  --output_dir results_meta_fomaml_run5 \
  --support_size 8 \
  --query_size 16 \
  --tasks_per_meta_batch 4 \
  --meta_steps 1000 \
  --inner_steps 1 \
  --inner_lr 3e-4 \
  --meta_lr 5e-5 \
  --trainable_mode last_k \
  --unfreeze_last_k 2 \
  --max_samples_per_group_train 1000 \
  --max_samples_per_group_val 500 \
  --max_samples_per_group_test 500

echo "Job completed successfully!"