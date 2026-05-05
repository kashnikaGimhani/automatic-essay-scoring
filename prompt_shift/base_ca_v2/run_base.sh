#!/bin/bash
#SBATCH --job-name=BASE_AES_CROSS_ATTENTION
#SBATCH -o base_aes_cross_attn.out
#SBATCH -e base_aes_cross_attn.err
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

python3 base_ca_v2/base.py \
  --data_path ../data/asap_train_with_all_traits.tsv \
  --output_dir outputs_bca_v2/base_prompt2 \
  --prompt_meta_json ../data/asap_prompt_meta_v3.json \
  --heldout_prompt 2 \
  --sep $'\t' \
  --num_epochs 5 \
  --patience 2 \
  --batch_size 2 \
  --eval_batch_size 8 \
  --max_length 480 \
  --rubric_max_length 192 \
  --rubric_model_name roberta-base \
  --rubric_text_field cross_attention_score_texts \
  --cross_attn_heads 8 \
  --fusion_init 0.1 \
  --pooling_type attention \
  --cross_attn_direction rubric_to_essay \
  --head_attn_heads 8 \
  --lr 2e-5 \
  --cross_attn_lr 1e-4 \
  --regressor_lr 1e-4 \
  --use_rubric_score_contrastive \
  --contrastive_weight 0.07 \
  --contrastive_temperature 0.07 \
  --score_rubric_text_field cross_attention_score_texts \
  --dev_ratio 0.1 \
  --save_split_files \
  --run_zero_shot_eval

# this script is for training the base model with cross attention and contrastive loss on rubric scores. head is still shared regression head.