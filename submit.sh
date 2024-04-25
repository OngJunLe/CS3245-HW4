#!/bin/bash
#SBATCH --time=300
#SBATCH --job-name=cs3245_hw4
#SBATCH --partition=medium
#SBATCH --mem=128G

# export PATH=$PATH:$HOME/miniconda3/bin
source $HOME/miniconda3/etc/profile.d/conda.sh
# eval "$('$HOME/miniconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate cs3245

free -m

#python index.py
python -u archive/index_dask.py -i 'data/dataset.csv' -d 'data/test' -p 'data/test'
