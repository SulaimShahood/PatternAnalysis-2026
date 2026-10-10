#!/bin/bash

#SBATCH --job-name=vqvae_hipmri
#SBATCH --partition=comp3710          
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_%x_%j.out

# Activate your conda environment
source $HOME/miniconda3/bin/activate
conda activate torch

MODEL=${1:-vqvae}

python train.py \
    --model "$MODEL" \
    --data-dir /home/groups/comp3710/HipMRI_Study_open/keras_slices_data \
    --epochs 30