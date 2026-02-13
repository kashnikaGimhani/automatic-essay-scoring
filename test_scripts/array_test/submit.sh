#!/bin/bash
#SBATCH --job-name=test_folds
#SBATCH --array=1-5
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:10:00
#SBATCH --output=logs/fold_%a.out
#SBATCH --error=logs/fold_%a.err

# Ensure log directory exists
mkdir -p logs

# Read the array task ID
FOLD_ID=${SLURM_ARRAY_TASK_ID}

echo "=============================="
echo "Running fold ${FOLD_ID}"
echo "=============================="

# Run the test script
python3 test.py --fold ${FOLD_ID}
