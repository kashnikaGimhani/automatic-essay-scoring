#!/bin/bash
#SBATCH --job-name=BASE_DEBERTA_ORD
#SBATCH -o base_deberta_ordinal.out
#SBATCH -e base_deberta_ordinal.err
#SBATCH --partition=bigmem
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sarathka@staff.vuw.ac.nz

set -euo pipefail
set -x
trap 'echo "Error at line $LINENO"; exit 1' ERR

module load GCCcore/11.2.0
module load Python/3.9.6

source mytest/bin/activate
export PYTHONUNBUFFERED=1

python3 deberta/base.py \
  --data_path ../data/asap_train_with_all_traits.tsv \
  --output_root outputs_deberta_ordinal \
  --heldout_prompts 2 \
  --encoder_name microsoft/deberta-v3-base \
  --head_type ordinal \
  --num_bins 6 \
  --pooling cls \
  --max_length 512 \
  --batch_size 4 \
  --eval_batch_size 8 \
  --num_epochs 8 \
  --lr 2e-5 \
  --head_lr 1e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --grad_accum_steps 1 \
  --patience 3 \
  --sep $'\t'
