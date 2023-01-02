#!/bin/bash
#SBATCH -J msa
#SBATCH -t 5:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=msa.out
#SBATCH --error=msa.err

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

cd /nas/longleaf/home/mixarcid/enzyme-stability

python get_msa.py
