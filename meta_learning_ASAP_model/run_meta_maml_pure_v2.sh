#!/bin/bash

#SBATCH --job-name=META_MAML_PURE
#SBATCH -o meta_maml_pure.out
#SBATCH -e meta_maml_pure.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sarathka@staff.vuw.ac.nz

set -euo pipefail
set -x

module purge
module load GCCcore/11.2.0
module load Python/3.9.6

source mytest/bin/activate
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Node: $(hostname)"
module list
which python
python --version
nvidia-smi

python3 -u meta_maml_pure_v2.py \
  --source_csv data_v2/asap_train_with_all_traits.tsv \
  --target_csv data_v2/113_copy.tsv \
  --output_dir results_meta_maml_pure_safe \
  --model_name "google/flan-t5-base" \
  --source_prompt_col "essay_set" \
  --target_prompt_col "essay_set" \
  --source_essay_col "essay" \
  --target_essay_col "essay" \
  --source_trait_map data_v2/source_trait_map.json \
  --target_trait_map data_v2/target_trait_map.json \
  --source_prompt_map data_v2/source_prompt_map.json \
  --target_prompt_map data_v2/target_prompt_map.json \
  --seed 42 \
  --max_input_length 256 \
  --max_target_length 16 \
  --generation_max_new_tokens 12 \
  --score_token_weight 4.0 \
  --separator_token_weight 1.0 \
  --constrained_num_beams 2 \
  --warmup_epochs 0 \
  --warmup_lr 1e-4 \
  --warmup_batch_size 4 \
  --warmup_dev_frac 0.10 \
  --meta_steps 200 \
  --meta_batch_size 1 \
  --support_size 4 \
  --query_size 8 \
  --inner_lr 1e-3 \
  --inner_steps 1 \
  --outer_lr 3e-5 \
  --weight_decay 1e-2 \
  --n_trainable_encoder_blocks 1 \
  --n_trainable_decoder_blocks 1 \
  --train_final_layer_norms \
  --support_frac 0.60 \
  --dev_frac 0.20 \
  --adapt_epochs 15 \
  --adapt_lr 5e-5 \
  --adapt_batch_size 4 \
  --prompt_sampling balanced \
  --meta_eval_every 25 \
  --meta_monitor_rows_per_prompt 24 \
  --meta_select_metric mean_trait_qwk \
  --debug_mode \
  --debug_examples 8