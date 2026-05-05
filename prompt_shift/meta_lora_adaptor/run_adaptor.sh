#!/bin/bash
#SBATCH --job-name=META_LORA_ADAPTOR
#SBATCH -o meta_lora_adaptor.out
#SBATCH -e meta_lora_adaptor.err
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

python3 meta_lora_adaptor/adaptor.py \
  --data_path ../data/asap_train_with_all_traits.tsv \
  --split_root target_splits \
  --base_root outputs \
  --output_root meta_lora_adaptor \
  --heldout_prompts 2 \
  --fewshot_sizes 8,16,32,64,128 \
  --max_length 512 \
  --loss_type mse \
  --meta_num_epochs 20 \
  --meta_episodes_per_epoch 40 \
  --meta_support_k 16 \
  --meta_query_k 32 \
  --meta_inner_steps 5 \
  --meta_inner_lr 5e-4 \
  --meta_step_size 0.1 \
  --adapt_num_epochs 30 \
  --adapt_lr 5e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --lora_target_modules query,value \
  --sep $'\t'