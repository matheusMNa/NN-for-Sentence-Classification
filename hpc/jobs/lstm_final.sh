#!/bin/bash
#SBATCH --job-name=final_lstm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/matheus25010/NN-for-Sentence-Classification
#SBATCH --output=hpc/logs/final_lstm_%j.out
#SBATCH --error=hpc/logs/final_lstm_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ilumpy

python -u experiments/final_scripts/final_lstm.py
