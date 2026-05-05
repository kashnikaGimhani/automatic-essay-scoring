#!/bin/bash
#SBATCH --job-name=CROSS_ATTENTION
#SBATCH -o cross_attention.out
#SBATCH -e cross_attention.err
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

python3 cross_attention/adaptor.py \
  --train_tsv ../data/asap_train_with_all_traits.tsv \
  --prompt_meta_json ../data/asap_prompt_meta_v2.json \
  --score_ranges_json ../data/asap_score_ranges.json \
  --heldout_prompt 2 \
  --k_shot 128 \
  --output_dir cross_attention/heldout_2 \
  --tasks_per_meta_batch 1 \
  --support_k 8 \
  --query_k 4 \
  --max_essay_length 384 \
  --max_meta_length 192 \
  --meta_epochs 20 \
  --unfreeze_top_n_layers 4 \
  --adapt_lr 1e-4 \
  --huber_delta 0.5