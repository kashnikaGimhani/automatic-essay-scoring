#!/bin/bash
#SBATCH --job-name=LORA_CROSS_ATTENTION
#SBATCH -o lora_cross_attention.out
#SBATCH -e lora_cross_attention.err
#SBATCH --partition=bigmem
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sarathka@staff.vuw.ac.nz

set -euo pipefail
set -x

trap 'echo "Error at line $LINENO"; exit 1' ERR

module load GCCcore/11.2.0
module load Python/3.9.6

source mytest/bin/activate
export PYTHONUNBUFFERED=1

python3 lora_ca_v2/adaptor.py \
  --data_path ../data/asap_train_with_all_traits.tsv \
  --split_root target_splits_v2 \
  --base_root outputs_bca_v2 \
  --base_ckpt_prefix base_prompt \
  --output_root lora_cross_attention/run12 \
  --prompt_meta_json ../data/asap_prompt_meta_v3.json \
  --heldout_prompts 2 \
  --fewshot_sizes 128 \
  --dev_file_template dev_{k}.tsv \
  --loss_type mse \
  --num_epochs 15 \
  --patience 3 \
  --lr 5e-4 \
  --cross_attn_lr 1e-4 \
  --regressor_lr 5e-4 \
  --batch_size 4 \
  --eval_batch_size 8 \
  --max_length 480 \
  --rubric_max_length 192 \
  --rubric_model_name roberta-base \
  --rubric_text_field cross_attention_score_texts \
  --score_rubric_text_field cross_attention_score_texts \
  --contrastive_weight 0.2 \
  --contrastive_temperature 0.07 \
  --cross_attn_heads 8 \
  --cross_attn_direction auto \
  --fusion_init 0.3 \
  --pooling_type auto \
  --head_attn_heads 8 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --lora_target_modules query,value,key \
  --sep $'\t'

# --use_rubric_score_contrastive \
# --freeze_shared_regressor \
# adaptor script for contrastive loss initial stable version
