#!/bin/bash
#SBATCH --job-name=ballet-misc
#SBATCH --output=results/logs/misclassified_%j.out
#SBATCH --error=results/logs/misclassified_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

source /gpfsnyu/spack/opt/spack/linux-rhel8-icelake/gcc-8.5.0/anaconda3-2023.09-0-eeifkc2jbd4x2dqbyp67rrtuzixdekzs/etc/profile.d/conda.sh
conda activate mmpose-stable

cd $SLURM_SUBMIT_DIR

python src/find_misclassified.py

echo "Done"
