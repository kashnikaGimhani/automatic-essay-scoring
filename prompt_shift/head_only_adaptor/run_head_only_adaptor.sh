#!/bin/bash
#SBATCH --job-name=HEAD_ONLY_ADAPTOR
#SBATCH -o head_only_adaptor.out
#SBATCH -e head_only_adaptor.err
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

python3 head_only_adaptor/head_only_adaptor.py \
  --data_path your_data.tsv \
  --split_root experiments/target_splits \
  --base_root experiments/base_models \
  --output_root experiments/head_only \
  --heldout_prompts all \
  --fewshot_sizes 8,16,32,64,128 \
  --loss_type mse \
  --num_epochs 30 \
  --lr 1e-3 \
  --batch_size 4 \
  --eval_batch_size 8 \
  --sep $'\t'