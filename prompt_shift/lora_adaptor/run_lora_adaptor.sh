#!/bin/bash
#SBATCH --job-name=LORA_ADAPTOR
#SBATCH -o lora_adaptor.out
#SBATCH -e lora_adaptor.err
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

python3 lora_adaptor/lora_adaptor.py \
  --data_path ../data/asap_train_with_all_traits.tsv \
  --split_root target_splits \
  --base_root outputs \
  --output_root lora_adaptor/run1 \
  --heldout_prompts 2 \
  --fewshot_sizes 32,64,128 \
  --loss_type mse \
  --num_epochs 30 \
  --lr 5e-4 \
  --batch_size 4 \
  --eval_batch_size 8 \
  --max_length 480 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --lora_target_modules query,value \
  --sep $'\t'