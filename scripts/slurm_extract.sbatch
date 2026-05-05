#!/bin/bash
#SBATCH --job-name=ballet-extract
#SBATCH --output=results/logs/extract_%A_%a.out
#SBATCH --error=results/logs/extract_%A_%a.err
#SBATCH --array=0-2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

SPLITS=(train val test)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

echo "Processing split: $SPLIT"
echo "Job ID: $SLURM_JOB_ID, Array task: $SLURM_ARRAY_TASK_ID"

source /gpfsnyu/spack/opt/spack/linux-rhel8-icelake/gcc-8.5.0/anaconda3-2023.09-0-eeifkc2jbd4x2dqbyp67rrtuzixdekzs/etc/profile.d/conda.sh
conda activate mmpose-stable

cd $SLURM_SUBMIT_DIR

python src/extract_poses.py --config config.yaml --split $SPLIT

echo "Done: $SPLIT"
