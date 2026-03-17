#!/bin/bash
#
#SBATCH --job-name=AES_Lora_Finetune
#SBATCH -o lora_finetune.out
#SBATCH -e lora_finetune.err
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

python3 model_finetune_lora.py \
    --data_path "./data/vuw_data_folds/fold_1" \
    --base_model "./results/fold_1/checkpoint-10000" \
    --output_path "./results_small_lora_asaptraits/fold_1"

#pip install -U transformers datasets accelerate peft sentencepiece scikit-learn pandas