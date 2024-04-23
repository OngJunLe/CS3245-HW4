#!/bin/bash
#SBATCH --time=300
#SBATCH --job-name=cs3245_hw4
#SBATCH --partition=medium
#SBATCH --mem=200GB

source $HOME/miniconda3/etc/profile.d/conda.sh
#bash
conda activate cs3245

free -m

ifconfig -a

#python index.py
python -u index_plsa_dask.py
#python query_processor.py