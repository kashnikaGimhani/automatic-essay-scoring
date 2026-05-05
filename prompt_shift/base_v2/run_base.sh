#!/bin/bash
#SBATCH --job-name=BASE_AES_V2
#SBATCH -o base_aes_v2.out
#SBATCH -e base_aes_v2.err
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

python3 base_v2/base.py \
  --data_path ../data/asap_train_with_all_traits.tsv \
  --output_dir outputs_v2/base_prompt2 \
  --heldout_prompt 2 \
  --source_prompts 1 \
  --sep $'\t' \
  --num_epochs 10 \
  --batch_size 4 \
  --eval_batch_size 8 \
  --max_length 480 \
  --dev_ratio 0.1 \
  --save_split_files \
  --run_zero_shot_eval