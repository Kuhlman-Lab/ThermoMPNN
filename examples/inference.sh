#!/bin/bash
#SBATCH -J inference-combo-test
#SBATCH --time 8:00:00
#SBATCH --partition=volta-gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=jobs/R-%x.%j.out
#SBATCH --error=jobs/R-%x.%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access

source ~/.bashrc
source /nas/longleaf/home/oem/.virtualenvs/ThermoMPNN/bin/activate

module load gcc
module load cuda

repo_location="../analysis"

cd $repo_location

python thermompnn_benchmarking.py --keep_preds
