#!/bin/bash
#SBATCH --job-name=ballet-extract
#SBATCH --output=results/logs/extract_%A_%a.out
#SBATCH --error=results/logs/extract_%A_%a.err
#SBATCH --array=0-23          # 8 chunks x 3 splits = 24 jobs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00

# Map array task ID to split and chunk
# Tasks 0-7:  train chunks 0-7
# Tasks 8-15: val chunks 0-7
# Tasks 16-23: test chunks 0-7
TOTAL_CHUNKS=8
SPLIT_IDX=$((SLURM_ARRAY_TASK_ID / TOTAL_CHUNKS))
CHUNK=$((SLURM_ARRAY_TASK_ID % TOTAL_CHUNKS))
SPLITS=(train val test)
SPLIT=${SPLITS[$SPLIT_IDX]}

echo "Split: $SPLIT | Chunk: $CHUNK/$TOTAL_CHUNKS"
echo "Job ID: $SLURM_JOB_ID, Array task: $SLURM_ARRAY_TASK_ID"

source /gpfsnyu/spack/opt/spack/linux-rhel8-icelake/gcc-8.5.0/anaconda3-2023.09-0-eeifkc2jbd4x2dqbyp67rrtuzixdekzs/etc/profile.d/conda.sh
conda activate mmpose-stable

cd $SLURM_SUBMIT_DIR

python src/extract_poses.py \
    --config config.yaml \
    --split $SPLIT \
    --chunk $CHUNK \
    --total-chunks $TOTAL_CHUNKS

echo "Done: $SPLIT chunk $CHUNK"
