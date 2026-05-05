#!/bin/bash
#SBATCH --job-name=ballet-train
#SBATCH --output=results/logs/train_%j.out
#SBATCH --error=results/logs/train_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

EXP=${1:-${EXP:-exp1}}

echo "Training experiment: $EXP"
echo "Job ID: $SLURM_JOB_ID"

cd $SLURM_SUBMIT_DIR

python src/train.py --config config.yaml --exp $EXP

echo "Done: $EXP"
