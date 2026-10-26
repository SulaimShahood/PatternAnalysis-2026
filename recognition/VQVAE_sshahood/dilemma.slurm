#!/bin/bash
#SBATCH --job-name=dilemma_hipmri
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

source $HOME/miniconda3/bin/activate
conda activate torch

python dilemma.py
