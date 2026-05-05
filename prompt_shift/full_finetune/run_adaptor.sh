#!/bin/bash
#SBATCH --job-name=FULL_FINETUNE
#SBATCH -o full_ft.out
#SBATCH -e full_ft.err
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

python3 full_finetune/adaptor.py \
  --data_path ../data/asap_train_with_all_traits.tsv \
  --split_root target_splits \
  --base_root outputs \
  --output_root full_ft \
  --heldout_prompts 2 \
  --fewshot_sizes 128 \
  --loss_type mse \
  --num_epochs 30 \
  --lr 2e-5 \
  --batch_size 4 \
  --eval_batch_size 8 \
  --max_length 512 \
  --sep $'\t'