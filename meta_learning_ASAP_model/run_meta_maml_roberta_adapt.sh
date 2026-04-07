#!/bin/bash

#SBATCH --job-name=META_MAML_ROBERTA_ADAPT
#SBATCH -o meta_maml_roberta_adapt.out
#SBATCH -e meta_maml_roberta_adapt.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sarathka@staff.vuw.ac.nz

set -euo pipefail
set -x

trap 'echo "Error at line $LINENO"; exit 1' ERR

module load GCCcore/11.2.0
module load Python/3.9.6

source mytest/bin/activate
export PYTHONUNBUFFERED=1

MODEL_NAME="roberta-base"
MAX_LENGTH=256
DROPOUT=0.3
LORA_R=4
LORA_ALPHA=16
LORA_DROPOUT=0.1
SEED=42
FREEZE_BASE="--freeze_base"
META_EPOCHS=4
META_STEPS_PER_EPOCH=250
META_PER_PROMPT_BATCH=8
META_LR=1e-5
META_INNER_LR=1e-3
META_TRAIN_RATIO=0.75
META_TEST_WEIGHT=1.0


TARGET_RANGES='{"ideas":[1,6],"flow":[1,6],"coherence":[1,6],"vocab":[1,6],"grammar":[1,6]}'

ADAPT_EPOCHS=30
ADAPT_BATCH_SIZE=8
ADAPT_LR=1e-4
ADAPT_STEPS_PER_EPOCH=50
ADAPT_PER_PROMPT_BATCH=4
ADAPT_INNER_LR=1e-3
ADAPT_META_TRAIN_RATIO=0.5
ADAPT_META_TEST_WEIGHT=1.0


python3 -u meta_maml_roberta.py adapt \
    --source_checkpoint results_meta_maml_roberta/train/best.pt \
    --target_file data_v2/113_copy.tsv \
    --target_train_ratio 0.6 \
    --target_dev_ratio 0.2 \
    --target_test_ratio 0.2 \
    --split_strategy prompt_overall \
    --split_num_bins 3 \
    --target_trait_ranges "$TARGET_RANGES" \
    --output_dir results_meta_maml_roberta/adapt \
    --model_name "$MODEL_NAME" \
    --max_length "$MAX_LENGTH" \
    --dropout "$DROPOUT" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --seed "$SEED" \
    --adapt_mode standard \
    --adapt_epochs "$ADAPT_EPOCHS" \
    --batch_size "$ADAPT_BATCH_SIZE" \
    --adapt_lr "$ADAPT_LR" \
    $FREEZE_BASE