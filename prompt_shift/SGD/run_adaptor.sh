#!/bin/bash
#SBATCH --job-name=META_SGD_RUN3
#SBATCH -o meta_sgd_run3.out
#SBATCH -e meta_sgd_run3.err
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 SGD/adaptor.py \
  --data_path ../data/asap_train_with_all_traits.tsv \
  --split_root target_splits \
  --base_root outputs \
  --output_root meta_sgd \
  --heldout_prompts 2 \
  --fewshot_sizes 16,128 \
  --max_length 512 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --loss_type mse \
  --meta_method meta_sgd \
  --meta_num_epochs 20 \
  --meta_episodes_per_epoch 40 \
  --meta_support_k 16 \
  --meta_query_k 32 \
  --meta_inner_steps 1 \
  --meta_outer_lr 1e-4 \
  --meta_sgd_init_alpha 1e-3 \
  --target_adapt_method meta_sgd \
  --target_meta_sgd_steps 1 \
  --target_meta_sgd_batch_size 1 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --lora_target_modules query,key,value \
  --lora_layers 6,7,8,9,10,11 \
  --lora_layers_pattern layer \
  --meta_first_order \
  --sep $'\t'
