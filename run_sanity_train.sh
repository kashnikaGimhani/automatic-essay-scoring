#!/bin/bash
set -e

# (Optional) activate virtual environment
# source aes_env/bin/activate

python3 model_trainer.py \
  --train_batch 2 \
  --valid_batch 2 \
  --test_batch 2 \
  --epoch 1 \
  --max_tgt_len 64 \
  --output_path "./results/fold_1" \
  --data_path "data/5fold_cv/fold_1" \
  --steps 50 \
  --model t5-small
