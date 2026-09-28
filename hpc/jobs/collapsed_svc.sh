#!/bin/bash
#SBATCH --job-name=predict_svc
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --chdir=/home/matheus25010/NN-for-Sentence-Classification
#SBATCH --output=hpc/logs/collapsed_svc_%j.out
#SBATCH --error=hpc/logs/collapsed_svc_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ilumpy

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python -u experiments/final_scripts/predict_collapse_svc.py
