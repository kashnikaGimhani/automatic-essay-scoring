#!/bin/bash

#SBATCH --job-name=TEST_GPU
#SBATCH -o test_gpu.out
#SBATCH -e test_gpu.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sarathka@staff.vuw.ac.nz

set -euo pipefail
set -x

trap 'echo "Error at line $LINENO"; exit 1' ERR

module load GCCcore/11.2.0
module load Python/3.9.6



source mytest/bin/activate
export PYTHONUNBUFFERED=1

echo "Node: $(hostname)"
nvidia-smi

python3 -u test_gpu.py