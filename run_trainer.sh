#!/bin/bash
#
#SBATCH --job-name=AES_Trainer
#SBATCH -o python_test.out
#SBATCH -e python_test.err
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sarathka@staff.vuw.ac.nz

module load GCCcore/11.2.0
module load Python/3.9.6

source mytest/bin/activate

python3 model_trainer.py \
  --train_batch 4 \
  --valid_batch 4 \
  --test_batch 4 \
  --epoch 15 \
  --max_tgt_len 166 \
  --output_path "./results/fold_1" \
  --data_path "data/5fold_cv/fold_1" \
  --steps 5000 \
  --model t5-large