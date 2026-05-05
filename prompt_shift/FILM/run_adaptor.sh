#!/bin/bash
#SBATCH --job-name=FILM_ADAPTOR
#SBATCH -o film_adaptor.out
#SBATCH -e film_adaptor.err
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

python3 FILM/adaptor.py \
  --data_path ../data/asap_train_with_all_traits.tsv \
  --heldout_prompt 2 \
  --output_dir FILM/heldout_2 \
  --model_name roberta-base \
  --epochs 5 \
  --batch_size 8 \
  --exclude_columns target overall \
  --prompt_meta_json ../data/asap_prompt_meta.json \
  --score_range_json ../data/asap_score_ranges.json \
  --match_loss_weight 0.1