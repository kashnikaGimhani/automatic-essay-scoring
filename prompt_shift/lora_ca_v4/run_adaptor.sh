#!/bin/bash
#SBATCH --job-name=LORA_RUBRIC_ENCODER
#SBATCH -o lora_rubric_encoder.out
#SBATCH -e lora_rubric_encoder.err
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 lora_ca_v4/adaptor.py \
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
  --lr 2e-4 \
  --cross_attn_lr 1e-4 \
  --regressor_lr 2e-4 \
  --batch_size 2 \
  --grad_accum_steps 2 \
  --eval_batch_size 8 \
  --max_length 480 \
  --rubric_max_length 192 \
  --rubric_model_name roberta-base \
  --rubric_text_field cross_attention_score_texts \
  --score_rubric_text_field cross_attention_score_texts \
  --use_rubric_score_contrastive \
  --contrastive_weight 0.05 \
  --contrastive_temperature 0.07 \
  --score_rubric_encode_batch_size 8 \
  --freeze_score_rubric_encoder_in_contrastive \
  --cross_attn_heads 8 \
  --cross_attn_direction auto \
  --fusion_init 0.5 \
  --pooling_type auto \
  --head_attn_heads 8 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.3 \
  --lora_target_modules query,value,key \
  --rubric_encoder_lora \
  --rubric_lora_r 8 \
  --rubric_lora_alpha 16 \
  --rubric_lora_dropout 0.3 \
  --rubric_lora_target_modules query,value,key \
  --rubric_lora_lr 2e-5 \
  --rubric_encoder_gradient_checkpointing \
  --sep $'\t'

# adaptor script for contrastive loss initial stable version

# Updated version: adds LoRA adaptation to the rubric encoder during few-shot training.
