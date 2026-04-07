#!/bin/bash

#SBATCH --job-name=META_ANIL_LORA
#SBATCH -o meta_anil_lora.out
#SBATCH -e meta_anil_lora.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=03:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sarathka@staff.vuw.ac.nz

set -euo pipefail
set -x

trap 'echo "Error at line $LINENO"; exit 1' ERR

module load GCCcore/11.2.0
module load Python/3.9.6

source mytest/bin/activate
export PYTHONUNBUFFERED=1

python3 meta_anil_lora.py \
  --source_csv data_v2/asap_train_with_all_traits.tsv \
  --target_csv data_v2/113_copy.tsv \
  --output_dir results_meta_anil_lora \
  --model_name "google/flan-t5-base" \
  --prompt_col "essay_set" \
  --essay_col "essay" \
  --overall_col "overall" \
  --source_trait_map data_v2/source_trait_map.json \
  --target_trait_map data_v2/target_trait_map.json \
  --max_length 512 \
  --seed 42 \
  --warmup_epochs 3 \
  --warmup_lr 2e-4 \
  --warmup_batch_size 8 \
  --meta_steps 200 \
  --meta_batch_size 2 \
  --support_size 8 \
  --query_size 16 \
  --inner_lr 1e-2 \
  --inner_steps 1 \
  --outer_lr 5e-5 \
  --weight_decay 1e-2 \
  --consistency_weight 0.25 \
  --overall_weight 1.0 \
  --n_trainable_blocks 2 \
  --support_frac 0.6 \
  --dev_frac 0.2 \
  --adapt_epochs 20 \
  --adapt_lr 5e-5 \
  --adapt_batch_size 4 \
  --eval_batch_size 8 \
  --target_use_lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1