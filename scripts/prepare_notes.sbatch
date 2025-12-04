#!/bin/bash
#SBATCH --time=5-00:00:00
#SBATCH --job-name=notes-mimic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
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
echo $(which python)

python PatientTrajectoryForecasting/prepare_notes.py \
    --clean_text \
    --clean_text \
    --lower \
    --remove_special_characters_mullenbach \
    --remove_special_characters \
    --remove_digits \
    --remove_accents \
    --remove_brackets \
    --convert_danish_characters \
    --apply_replace \
    --remove_adm_details \