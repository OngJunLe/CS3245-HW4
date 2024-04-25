#!/bin/bash
#SBATCH --time=30
#SBATCH --job-name=cs3245_hw4
#SBATCH --partition=normal

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate cs3245

free -m

# python -u query_processor.py
python search.py -d 'data/struct_compress_dictionary' -p 'data/struct_compress_postings' -q 'data/q3.txt' -o 'data/q3_output.txt'