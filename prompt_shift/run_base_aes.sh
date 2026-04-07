#!/bin/bash
#SBATCH --job-name=BASE_AES
#SBATCH -o base_aes.out
#SBATCH -e base_aes.err
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

python3 base_aes.py \
  --data_path ../data/asap_train_with_all_traits.tsv \
  --output_dir outputs/base_prompt2 \
  --heldout_prompt 2 \
  --sep $'\t' \
  --num_epochs 10 \
  --batch_size 4 \
  --eval_batch_size 8 \
  --max_length 768 \
  --dev_ratio 0.1 \
  --save_split_files \
  --run_zero_shot_eval