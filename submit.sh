#!/bin/bash
#SBATCH --time=300
#SBATCH --job-name=cs3245_hw4
#SBATCH --partition=medium
#SBATCH --mem=512G

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cs3245

free -m

#python index.py
python -u index.py -i 'data/dataset.csv' -d 'data/dictionary' -p 'data/postings'
#python query_processor.py