#!/bin/bash
# Slurm batch script for training on Rangpur.
# Usage: sbatch train.slurm vqvae       (or: sbatch train.slurm autoencoder)

#SBATCH --job-name=vqvae_hipmri
#SBATCH --partition=comp3710          # TODO: confirm the correct partition/QOS for this course on Rangpur
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_%x_%j.out

# TODO: activate whatever environment has torch/nibabel/scikit-image installed, e.g.:
# source /home/Student/<your_student_id>/miniconda3/etc/profile.d/conda.sh
# conda activate comp3710

MODEL=${1:-vqvae}

python train.py \
    --model "$MODEL" \
    --data-dir /home/groups/comp3710/HipMRI_Study_open/keras_slices_data \
    --epochs 30
