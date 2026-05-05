#!/bin/bash
#SBATCH --job-name=THRESHOLD_TUNE
#SBATCH -o threshold_tune.out
#SBATCH -e threshold_tune.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sarathka@staff.vuw.ac.nz

set -euo pipefail
set -x

trap 'echo "Error at line $LINENO"; exit 1' ERR

module load GCCcore/11.2.0
module load Python/3.9.6

source mytest/bin/activate
export PYTHONUNBUFFERED=1

python3 threshold_tune/adaptor_v2.py \
  --data_path ../data/asap_train_with_all_traits.tsv \
  --split_root target_splits_v2 \
  --base_root outputs \
  --head_only_root head_only_adaptor \
  --output_root threshold_tuned_head_only_v2 \
  --heldout_prompts 2 \
  --fewshot_sizes 8,16,32,64,128 \
  --dev_file_template 'dev_{k}.tsv' \
  --max_length 480 \
  --eval_batch_size 8 \
  --round_step 1.0 \
  --sep $'\t' \
  --include_zero_shot
