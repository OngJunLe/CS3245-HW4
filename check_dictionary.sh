#!/bin/bash
#SBATCH --time=30
#SBATCH --job-name=cs3245_hw4
#SBATCH --partition=standard

# export PATH=$PATH:$HOME/miniconda3/bin
source $HOME/miniconda3/etc/profile.d/conda.sh
# eval "$('$HOME/miniconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate cs3245

free -m

python check_dictionary.py
# python -u index.py -i 'data/dataset.csv' -d 'data/struct_dictionary' -p 'data/struct_postings'
#python query_processor.py