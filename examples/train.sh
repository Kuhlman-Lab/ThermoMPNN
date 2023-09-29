#!/bin/bash
#SBATCH -J train-combo-no-hidden-dims
#SBATCH -t 3-00:00:00
#SBATCH --partition=volta-gpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=jobs/R-%x.%j.out
#SBATCH --error=jobs/R-%x.%j.err
#SBATCH --qos=gpu_access

source ~/.bashrc
source /nas/longleaf/home/oem/.virtualenvs/ThermoMPNN/bin/activate

module load gcc
module load cuda

repo_location="../ThermoMPNN/"

cd $repo_location

python train_thermompnn.py config.yaml $@
