#!/bin/bash
#SBATCH --job-name=HEAD_ONLY_DEBERTA_ORD
#SBATCH -o head_only_deberta_ordinal.out
#SBATCH -e head_only_deberta_ordinal.err
#SBATCH --partition=bigmem
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

python3 deberta/adaptor.py \
  --data_path ../data/asap_train_with_all_traits.tsv \
  --split_root target_splits \
  --base_root outputs_deberta_ordinal \
  --output_root head_only_deberta_ordinal \
  --heldout_prompts 2 \
  --fewshot_sizes 32,64,128 \
  --encoder_name microsoft/deberta-v3-base \
  --head_type ordinal \
  --num_bins 6 \
  --pooling cls \
  --max_length 512 \
  --num_epochs 30 \
  --lr 1e-3 \
  --batch_size 4 \
  --eval_batch_size 8 \
  --sep $'\t'
