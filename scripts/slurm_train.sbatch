#!/bin/bash
#SBATCH --job-name=ballet-train
#SBATCH --output=results/logs/train_%j.out
#SBATCH --error=results/logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

EXP=${1:-${EXP:-exp1}}

echo "Training experiment: $EXP"
echo "Job ID: $SLURM_JOB_ID"

source /gpfsnyu/spack/opt/spack/linux-rhel8-icelake/gcc-8.5.0/anaconda3-2023.09-0-eeifkc2jbd4x2dqbyp67rrtuzixdekzs/etc/profile.d/conda.sh
conda activate mmpose-stable

cd $SLURM_SUBMIT_DIR

python src/train.py --config config.yaml --exp $EXP

echo "Done: $EXP"
