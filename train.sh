#!/bin/bash
#SBATCH -J train
#SBATCH -t 10-00:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=jobs/R-%x.%j.out
#SBATCH --error=jobs/R-%x.%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access
#SBATCH --exclude=g0601

__conda_setup="$('/nas/longleaf/home/mixarcid/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/nas/longleaf/home/mixarcid/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/nas/longleaf/home/mixarcid/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/nas/longleaf/home/mixarcid/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

module load gcc/11.2.0
module load cuda/11.8
conda activate chem-py3.9

cd /nas/longleaf/home/mixarcid/enzyme-stability

# python train.py $@

python train.py loss.ddG_lambda=0.1 loss.dTm_lambda=0.9 name=combo_0.1_0.9
python train.py loss.ddG_lambda=0.5 loss.dTm_lambda=0.5 name=combo_0.0.5_0.5
python train.py loss.ddG_lambda=0.9 loss.dTm_lambda=0.1 name=combo_0.9_0.1
python train.py datasets=[rocklin] name=rocklin_ml