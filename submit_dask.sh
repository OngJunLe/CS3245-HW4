#!/bin/bash
#SBATCH --time=300
#SBATCH --job-name=cs3245_hw4
#SBATCH --partition=medium
#SBATCH --mem=128G

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cs3245

free -m

ifconfig -a

#python index.py
python -u index_plsa_dask.py
#python query_processor.py