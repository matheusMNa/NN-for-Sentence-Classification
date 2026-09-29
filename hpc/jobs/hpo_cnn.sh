#!/bin/bash
#SBATCH --job-name=hpo_cnn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/matheus25010/NN-for-Sentence-Classification
#SBATCH --output=hpc/logs/hpo_cnn_%j.out
#SBATCH --error=hpc/logs/hpo_cnn_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ilumpy

export N_TRIALS=50

python -u experiments/optimization_scripts/hpo_cnn.py
