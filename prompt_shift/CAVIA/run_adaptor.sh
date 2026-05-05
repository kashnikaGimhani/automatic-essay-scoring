#!/bin/bash
#SBATCH --job-name=CAVIA_RUN2
#SBATCH -o cavia_run2.out
#SBATCH -e cavia_run2.err
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

python3 CAVIA/adaptor.py \
  --data_path ../data/asap_train_with_all_traits.tsv \
  --sep '\t' \
  --prompt_col essay_set \
  --text_col essay \
  --heldout_prompt 2 \
  --score_ranges_json ../data/asap_score_ranges.json \
  --prompt_specs_json ../data/asap_prompt_meta_v2.json \
  --output_dir results/cavia_rubric_heldout2 \
  --encoder_name roberta-base \
  --max_length 384 \
  --meta_steps 1000 \
  --meta_batch_tasks 1 \
  --support_size 8 \
  --query_size 8 \
  --inner_steps 3 \
  --inner_lr 0.1 \
  --context_dim 32 \
  --repeats 5 \
  --k_values 8 16 32 64 128 \
  --first_order \
  --gradient_checkpointing \
  --amp_dtype bf16
