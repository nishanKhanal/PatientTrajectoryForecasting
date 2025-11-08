#!/bin/bash
#SBATCH --time=5-00:00:00
#SBATCH --job-name=notes-mimic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=2
#SBATCH --output=logs/sbatch/prepare-notes/try.%j
#SBATCH --mem=64G
#SBATCH --partition=gpu


# Load necessary modules (adjust as needed for your system)
ml miniconda3
ml cuda/11.8

# echo running conda init
# conda init 
# echo ran conda init

source ~/.bashrc
# Activate a specific conda environment 
conda activate $kdd_proj_env

python 