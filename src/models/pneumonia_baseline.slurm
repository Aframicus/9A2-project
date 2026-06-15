#!/bin/bash
#SBATCH --job-name=pneumonia_baseline
#SBATCH --partition=rome
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --output=%x-%j.out 

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

source ~/venv/bin/activate
cd ~/9A2-project/src/models
python -u pneumonia_baseline.py