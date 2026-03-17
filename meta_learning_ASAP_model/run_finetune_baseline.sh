#!/bin/bash
#
#SBATCH --job-name=AES_Finetune_Baseline
#SBATCH -o finetune_baseline.out
#SBATCH -e finetune_baseline.err
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

python3 finetune_baseline_v2.py \
  --train_file ../data/asap_train_with_all_traits.tsv \
  --model_path ../results/fold_1/checkpoint-10000 \
  --output_dir ./results_finetune_baseline \
  --num_epochs 2 \
  --train_batch_size 4 \
  --eval_batch_size 4 \
  --learning_rate 2e-5 \
  --max_samples_per_group_train 300 \
  --max_samples_per_group_val 100 \
  --train_prompts 1 2 3 4 5 6 \
  --val_prompts 7 \
  --test_prompts 8