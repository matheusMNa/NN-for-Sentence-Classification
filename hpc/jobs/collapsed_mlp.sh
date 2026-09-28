#!/bin/bash
#SBATCH --job-name=predict_mlp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --chdir=/home/matheus25010/NN-for-Sentence-Classification
#SBATCH --output=hpc/logs/collapsed_mlp_%j.out
#SBATCH --error=hpc/logs/collapsed_mlp_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ilumpy

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python -u experiments/final_scripts/predict_collapse_mlp.py
