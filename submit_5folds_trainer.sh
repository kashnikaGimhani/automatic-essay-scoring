#!/bin/bash
#
#SBATCH --job-name=AES_Trainer_5folds
#SBATCH --output=logs/fold_%a.out
#SBATCH --error=logs/fold_%a.err
#
#SBATCH --array=1-5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=23:00:00
#
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sarathka@staff.vuw.ac.nz

mkdir -p sbatch_logs

module load GCCcore/11.2.0
module load Python/3.9.6

source mytest/bin/activate

FOLD_ID=${SLURM_ARRAY_TASK_ID}

DATA_PATH="data/5fold_cv/fold_${FOLD_ID}"
OUTPUT_PATH="./results/fold_${FOLD_ID}"

echo "======================================"
echo "Running AES training for fold ${FOLD_ID}"
echo "Data path:   ${DATA_PATH}"
echo "Output path: ${OUTPUT_PATH}"
echo "======================================"

python3 model_trainer.py \
  --train_batch 4 \
  --valid_batch 4 \
  --test_batch 4 \
  --epoch 15 \
  --max_tgt_len 166 \
  --output_path "${OUTPUT_PATH}" \
  --data_path "${DATA_PATH}" \
  --steps 5000 \
  --model t5-large