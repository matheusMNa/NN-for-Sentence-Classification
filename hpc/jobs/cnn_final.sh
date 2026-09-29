#!/bin/bash
#SBATCH --job-name=final_cnn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/matheus25010/NN-for-Sentence-Classification
#SBATCH --output=hpc/logs/final_cnn_%j.out
#SBATCH --error=hpc/logs/final_cnn_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ilumpy

echo "SLURM: $SLURM_CPUS_PER_TASK CPUs | visíveis ao processo: $(nproc)"

python -u experiments/final_scripts/final_cnn.py
