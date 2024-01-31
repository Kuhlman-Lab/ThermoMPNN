#!/bin/bash
#SBATCH -J inference
#SBATCH -t 30:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=jobs/R-%x.%j.out
#SBATCH --error=jobs/R-%x.%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access

source ~/.bashrc
conda activate thermoMPNN

module load gcc
module load cuda

repo_location="/proj/kuhl_lab/ThermoMPNN/analysis"

cd $repo_location

python custom_inference.py --pdb ../examples/2OCJ.pdb --chain A --model_path ../models/thermoMPNN_default.pt
